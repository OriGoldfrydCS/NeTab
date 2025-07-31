import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    GPT2Config
)
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


class NetworkTrafficModel(nn.Module):
    """
    AI model for generating synthetic network traffic data using GPT-2 architecture
    
    Supports two training modes:
    - Mode 0: Frozen base model with trainable MLP head 
    - Mode 1: Full fine-tuning of the entire model 
    
    Features:
    - Tokenizes 90 PPI features + 1 APP label into special tokens
    - Uses logarithmic binning for numerical value quantization
    - Supports both training and generation of network traffic patterns
    """
    
    def __init__(
        self, 
        model_name: str = "gpt2",         # Base model to use (GPT-2 variants)
        training_mode: int = 0,           # 0=frozen base+MLP, 1=full fine-tuning
        vocab_size: int = 50257,          # Original vocabulary size
        n_features: int = 91,             # 90 PPI features + 1 APP label
        hidden_size: int = 768,           # Hidden dimension size
        freeze_base: bool = True          # Whether to freeze base model weights
    ):
        super().__init__()
        
        # Store configuration parameters
        self.model_name = model_name
        self.training_mode = training_mode
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Load tokenizer and set padding token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize base model based on training mode
        if training_mode == 0:
            # Mode 0: Frozen base + MLP head
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
            self._freeze_base_model()
            
        else:
            # Mode 1: Full fine-tuning
            self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Feature tokenization mappings (will be populated below)
        self.feature_to_token = {}
        self.token_to_feature = {}
        self._initialize_feature_mappings()
        
        # Initialize traffic head AFTER vocabulary extension (for mode 0 only)
        if training_mode == 0:
            # Custom MLP head for network traffic - uses extended vocabulary size
            extended_vocab_size = len(self.tokenizer)
            self.traffic_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, extended_vocab_size)
            )
        else:
            self.traffic_head = None
        
    def _freeze_base_model(self):
        """Freeze all parameters in the base model to prevent training"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _initialize_feature_mappings(self):
        """Initialize mappings between network features and special tokens"""
        # Create special tokens for structured data representation
        special_tokens = []
        
        # Add structural tokens for data organization
        special_tokens.extend([
            "<SAMPLE_START>", "<SAMPLE_END>", 
            "<PPI_START>", "<PPI_END>",
            "<APP_START>", "<APP_END>",
            "<SEP>"
        ])
        
        # Add PPI feature tokens (90 total: 3 directions × 30 features each)
        for i in range(3):  # PPI_0, PPI_1, PPI_2 (packet directions)
            for j in range(30):  # 0-29 (features per direction)
                special_tokens.append(f"<PPI_{i}_{j}>")
        
        # Add value tokens using logarithmic binning for better numerical coverage
        # Handle special cases first
        special_tokens.extend(["<VAL_NEG>", "<VAL_ZERO>"])  # For negative and zero values
        
        # Logarithmic bins for positive values (covers 1 to ~65,000 with 200 bins)
        import math
        max_val = 65536  # 2^16, covers our max observed value of 51,687
        num_bins = 200
        
        for i in range(num_bins):
            # Exponential spacing: 1, 2, 4, 8, 16, 32, ... (logarithmic distribution)
            bin_value = int(math.pow(2, i * math.log2(max_val) / num_bins))
            special_tokens.append(f"<VAL_{bin_value}>")
        
        # Add unknown value token for fallback cases
        special_tokens.append("<UNK_VAL>")
        
        # Store original vocab size before adding tokens
        original_vocab_size = len(self.tokenizer)
        
        # Add special tokens to tokenizer vocabulary
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Resize model embeddings to accommodate new tokens
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Create bidirectional mappings between tokens and IDs
        for token in special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.feature_to_token[token] = token_id
            self.token_to_feature[token_id] = token
            
        print(f"Extended vocabulary from {original_vocab_size} to {len(self.tokenizer)} tokens")
    
    def _quantize_value(self, value: float) -> str:
        """Convert a numerical value to the nearest logarithmic bin token"""
        if value < 0:
            return "<VAL_NEG>"
        elif value == 0.0:
            return "<VAL_ZERO>"
        else:
            # Find the closest logarithmic bin for positive values
            import math
            max_val = 65536
            num_bins = 200
            
            # Calculate which bin this value belongs to
            if value >= max_val:
                return f"<VAL_{max_val}>"
            
            # Map value to logarithmic bin index
            log_value = math.log2(max(1, value))
            bin_index = int(log_value * num_bins / math.log2(max_val))
            bin_index = max(0, min(num_bins - 1, bin_index))
            
            bin_value = int(math.pow(2, bin_index * math.log2(max_val) / num_bins))
            return f"<VAL_{bin_value}>"
    
    def encode_network_data(self, data_row: Dict[str, Any]) -> List[int]:
        """
        Encode a network traffic data row into token sequence
        
        Args:
            data_row: Dictionary containing PPI features and APP label
            
        Returns:
            List of token IDs representing the encoded data
        """
        tokens = []
        
        # Add sample start token
        tokens.append(self.feature_to_token["<SAMPLE_START>"])
        
        # Add PPI section start token
        tokens.append(self.feature_to_token["<PPI_START>"])
        
        # Encode all 90 PPI features systematically
        for i in range(3):  # 3 packet directions
            for j in range(30):  # 30 features per direction
                feature_name = f"PPI_{i}_{j}"
                if feature_name in data_row:
                    value = float(data_row[feature_name])
                    
                    # Add feature name token
                    feature_token = f"<PPI_{i}_{j}>"
                    tokens.append(self.feature_to_token[feature_token])
                    
                    # Add quantized value token
                    value_token = self._quantize_value(value)
                    
                    # Check if token exists, use unknown if not
                    if value_token in self.feature_to_token:
                        tokens.append(self.feature_to_token[value_token])
                    else:
                        tokens.append(self.feature_to_token["<UNK_VAL>"])
                    
                    # Add separator between feature-value pairs
                    tokens.append(self.feature_to_token["<SEP>"])
        
        # Add PPI section end token
        tokens.append(self.feature_to_token["<PPI_END>"])
        
        # Encode APP label section
        if "APP" in data_row:
            tokens.append(self.feature_to_token["<APP_START>"])
            # Encode APP name as regular text tokens
            app_tokens = self.tokenizer.encode(str(data_row["APP"]), add_special_tokens=False)
            tokens.extend(app_tokens)
            tokens.append(self.feature_to_token["<APP_END>"])
        
        # Add sample end token
        tokens.append(self.feature_to_token["<SAMPLE_END>"])
        
        return tokens
    
    def decode_network_data(self, tokens: List[int]) -> Dict[str, Any]:
        """
        Decode token sequence back to network traffic data
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Dictionary containing decoded PPI features and APP label
        """
        data_row = {}
        i = 0
        
        # Parse tokens sequentially
        while i < len(tokens):
            token_id = tokens[i]
            
            if token_id in self.token_to_feature:
                token_str = self.token_to_feature[token_id]
                
                # Handle PPI feature tokens
                if token_str.startswith("<PPI_"):
                    # Extract PPI feature name and value
                    feature_name = token_str[1:-1]  # Remove < and >
                    if i + 1 < len(tokens) and tokens[i + 1] in self.token_to_feature:
                        value_token = self.token_to_feature[tokens[i + 1]]
                        value = self._decode_value_token(value_token)
                        data_row[feature_name] = value
                        i += 2  # Skip both feature and value tokens
                        continue
                
                # Handle APP label section
                elif token_str == "<APP_START>":
                    # Extract APP label tokens
                    app_tokens = []
                    i += 1
                    while i < len(tokens):
                        if tokens[i] in self.token_to_feature and self.token_to_feature[tokens[i]] == "<APP_END>":
                            break
                        app_tokens.append(tokens[i])
                        i += 1
                    
                    # Decode APP name from tokens
                    if app_tokens:
                        app_name = self.tokenizer.decode(app_tokens, skip_special_tokens=True).strip()
                        data_row["APP"] = app_name
                    continue
            
            i += 1
        
        return data_row
    
    def _decode_value_token(self, value_token: str) -> float:
        """Decode a value token back to a numerical value"""
        if value_token == "<VAL_ZERO>":
            return 0.0
        elif value_token == "<VAL_NEG>":
            return -1.0  # Default negative value
        elif value_token == "<UNK_VAL>":
            return 0.0  # Default unknown values to 0
        elif value_token.startswith("<VAL_") and value_token.endswith(">"):
            try:
                value = float(value_token[5:-1])  # Extract number from <VAL_X>
                return value
            except ValueError:
                return 0.0
        else:
            return 0.0
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model based on training mode"""
        
        if self.training_mode == 0:
            # Mode 0: Use frozen base model + custom MLP head
            with torch.no_grad():  # Don't compute gradients for base model
                base_outputs = self.base_model.transformer(input_ids, attention_mask=attention_mask)
            
            # Pass through custom trainable head
            hidden_states = base_outputs.last_hidden_state
            logits = self.traffic_head(hidden_states)
            
        else:
            # Mode 1: Full model forward pass (all parameters trainable)
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': None
        }
    
    def generate_samples(
        self, 
        n_samples: int,                   # Number of samples to generate
        max_length: int = 512,            # Maximum sequence length
        temperature: float = 0.8,         # Sampling temperature (higher = more random)
        top_k: int = 50,                  # Top-k sampling parameter
        top_p: float = 0.9,               # Top-p (nucleus) sampling parameter
        device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device for inference
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic network traffic samples using the trained model
        
        Args:
            n_samples: Number of samples to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature (controls randomness)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            device: Device to run inference on
            
        Returns:
            List of generated network traffic data samples
        """
        self.to(device)
        self.eval()  # Set model to evaluation mode
        
        generated_samples = []
        
        with torch.no_grad():  # Disable gradient computation for inference
            for sample_idx in range(n_samples):
                if sample_idx % 10 == 0:
                    print(f"  Generating sample {sample_idx + 1}/{n_samples}...")
                
                # Use template-based approach to ensure complete samples
                sample = {}
                
                # Generate all 90 PPI features systematically
                for i in range(3):  # 3 packet directions
                    for j in range(30):  # 30 features per direction
                        ppi_name = f"PPI_{i}_{j}"
                        
                        # Create a prompt for this specific feature
                        prompt_tokens = [self.feature_to_token[f"<{ppi_name}>"]]
                        input_ids = torch.tensor([prompt_tokens], device=device)
                        
                        # Generate value for this feature using the model
                        if self.training_mode == 0:
                            # Mode 0: Use frozen base + custom head
                            base_outputs = self.base_model.transformer(input_ids)
                            logits = self.traffic_head(base_outputs.last_hidden_state)
                        else:
                            # Mode 1: Use full model
                            outputs = self.base_model(input_ids)
                            logits = outputs.logits
                        
                        # Apply temperature scaling for controlled randomness
                        next_token_logits = logits[0, -1, :] / temperature
                        
                        # Filter to only value tokens (VAL_*) for valid sampling
                        value_token_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        for token_name, token_id in self.feature_to_token.items():
                            if token_name.startswith('<VAL_'):
                                value_token_mask[token_id] = True
                        
                        # Set non-value tokens to very low probability
                        next_token_logits[~value_token_mask] = float('-inf')
                        
                        # Apply sampling with fallback handling
                        if torch.all(torch.isinf(next_token_logits)):
                            # Fallback: pick a random value token
                            value_tokens = [tid for name, tid in self.feature_to_token.items() if name.startswith('<VAL_')]
                            next_token = torch.tensor([np.random.choice(value_tokens)], device=device)
                        else:
                            # Apply top-k filtering if specified
                            if top_k > 0:
                                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, value_token_mask.sum()))[0][..., -1, None]
                                next_token_logits[indices_to_remove] = float('-inf')
                            
                            # Sample from filtered distribution
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Decode the sampled value token to numerical value
                        value_token_id = next_token.item()
                        value_token_name = None
                        for name, tid in self.feature_to_token.items():
                            if tid == value_token_id:
                                value_token_name = name
                                break
                        
                        # Convert token to numerical value
                        if value_token_name and value_token_name.startswith('<VAL_'):
                            # Extract numeric value from token name
                            if value_token_name == '<VAL_NEG>':
                                value = -1.0
                            elif value_token_name == '<VAL_ZERO>':
                                value = 0.0
                            elif value_token_name.startswith('<VAL_') and value_token_name.endswith('>'):
                                try:
                                    val_str = value_token_name[5:-1]  # Remove <VAL_ and >
                                    if val_str.isdigit():
                                        # For positive quantized values
                                        quantized_val = int(val_str)
                                        value = float(quantized_val)  # Use quantized value directly
                                    else:
                                        value = 0.0
                                except:
                                    value = 0.0
                            else:
                                value = 0.0
                        else:
                            value = 0.0
                        
                        sample[ppi_name] = value
                
                # Generate APP label (simplified approach)
                # Since we're training on specific app data, use that as default
                # In practice, this could be generated using sequence generation
                sample["APP"] = "Youtube"  # Default based on training data
                
                generated_samples.append(sample)
        
        return generated_samples
    
    def save_model(self, save_path: str):
        """Save the complete model and tokenizer to disk"""
        # Handle different saving approaches based on training mode
        if self.training_mode == 0:
            # For frozen base mode, temporarily enable all gradients to save full state
            original_requires_grad = {}
            for name, param in self.named_parameters():
                original_requires_grad[name] = param.requires_grad
                param.requires_grad = True
            
            # Get full state dict including frozen parameters
            full_state_dict = self.state_dict()
            
            # Restore original gradient requirements
            for name, param in self.named_parameters():
                param.requires_grad = original_requires_grad[name]
            
            model_state_dict = full_state_dict
        else:
            # For full fine-tuning mode, all parameters are trainable
            model_state_dict = self.state_dict()
        
        # Save model checkpoint with configuration
        torch.save({
            'model_state_dict': model_state_dict,
            'model_config': {
                'model_name': self.model_name,
                'training_mode': self.training_mode,
                'n_features': self.n_features,
                'hidden_size': self.hidden_size
            },
            'feature_mappings': {
                'feature_to_token': self.feature_to_token,
                'token_to_feature': self.token_to_feature
            }
        }, save_path)
        
        # Save tokenizer separately for easy loading
        self.tokenizer.save_pretrained(save_path + "_tokenizer")
    
    @classmethod
    def load_model(cls, load_path: str):
        """Load a pre-trained model from disk"""
        checkpoint = torch.load(load_path)
        config = checkpoint['model_config']
        
        # Create model instance with saved configuration
        model = cls(
            model_name=config['model_name'],
            training_mode=config['training_mode'],
            n_features=config['n_features'],
            hidden_size=config['hidden_size']
        )
        
        # Load model weights with error handling
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Could not load all model weights strictly. {e}")
            print("Attempting to load with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")           # Show first 5 for brevity
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")     # Show first 5 for brevity
        
        # Load feature mappings if available
        if 'feature_mappings' in checkpoint:
            model.feature_to_token = checkpoint['feature_mappings']['feature_to_token']
            model.token_to_feature = checkpoint['feature_mappings']['token_to_feature']
        
        # Load tokenizer with error handling
        try:
            model.tokenizer = AutoTokenizer.from_pretrained(load_path + "_tokenizer")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {load_path}_tokenizer: {e}")
            print("Using default tokenizer...")
        
        return model
