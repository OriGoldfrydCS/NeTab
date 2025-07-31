"""
Data-aware Network Traffic Generation with Precision-Guided Algorithm

WHAT THIS SCRIPT DOES:
This script creates artificial network traffic data that looks and behaves like real network traffic.
Think of it as an AI that learned from your real network data and can now create realistic
fake data for testing, research, or privacy protection.

HOW IT WORKS:
1. Analyzes your original network traffic data to understand patterns
2. Uses advanced AI-guided sampling to create new, realistic traffic samples
3. Applies smart refinement to ensure the generated data is high quality
4. Validates quality and saves results for you to use

THREE TYPES OF NETWORK DATA IT GENERATES:
- PPI_0_* (Inter-packet timing): How long between network packets (realistic timing)
- PPI_1_* (Packet direction): Whether packets go in/out (-1, 0, +1 values)  
- PPI_2_* (Packet sizes): How big each network packet is (realistic sizes)

WHAT YOU GET:
- High-quality synthetic network traffic data
- Quality analysis reports showing how good the generated data is
- Duplicate checking to ensure uniqueness from original data
- Detailed logs of the generation process
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import traceback
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# File paths - Update these to point to your files
ORIGINAL_DATA_PATH = "dataset/my_dataset.csv"       # Your real network traffic data
OUTPUT_DIR = "./generated_data"                     # Where synthetic data gets saved

# =============================================================================
# GENERATION SETTINGS - Control how much data to create
# =============================================================================

N_SAMPLES = 1000                                    # Number of synthetic samples to create
GENERATION_MODE = "data_aware_precision_guided"     # Smart distribution analysis + AI model + Algorithm refinement

# =============================================================================
# MODEL SETTINGS
# =============================================================================

MODEL_PATH = "trained_model/my_model.pt"            # Path to trained model file
USE_MODEL_GUIDANCE = True                           # Use AI model to improve quality

# Advanced settings - Usually don't need to change these
PERPLEXITY_THRESHOLD = 8.0           # How strict the AI model should be (how "surprised" it can be)
FORCE_REFINEMENT_PERCENTAGE = 0.15   # Percentage of samples to always improve (15% = moderate)

# Debugging settings
DEBUG_MODE = True       # Show detailed progress messages
LOG_FILE = None         # Will be set automatically when script runs

# =============================================================================
# LOGGING SYSTEM - Saves progress messages to screen and file
# =============================================================================

def log_message(message: str):
    """Save message to both screen and log file for troubleshooting"""
    print(message)  # Show on screen
    if LOG_FILE:    # Also save to file if log file is set
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

# =============================================================================
# DATA ANALYSIS FUNCTIONS - Understanding the original data
# =============================================================================
"""
These functions examine your original network traffic data to understand:
1. What patterns exist in the data
2. How different features relate to each other  
3. What ranges of values are realistic
4. How to generate new data that looks similar

The AI uses this analysis to create synthetic data that matches your original data's characteristics.
"""

def analyze_data_distributions(data: pd.DataFrame) -> dict:
    """
    Examine the original data to understand its patterns.
    This helps generate realistic synthetic data.
    """
    log_message("Analyzing data distributions and computing adaptive thresholds...")
    
    # Create analysis structure to store results
    analysis = {
        'ppi_groups': {},                           # Statistics for each PPI group
        'app_labels': list(data['APP'].unique()),   # List of application names
        'overall_stats': {},                        # General statistics
        'adaptive_thresholds': {}                   # Smart limits for each data type
    }
    
    # Analyze each of the 3 PPI groups separately
    for group in range(3):
        # Get column names for this group (each group has 30 columns)
        group_cols = [f"PPI_{group}_{i}" for i in range(30)]
        group_data = data[group_cols]
        
        # Calculate basic statistics
        min_val = float(group_data.min().min())     # Smallest value
        max_val = float(group_data.max().max())     # Largest value
        mean_val = float(group_data.mean().mean())  # Average value
        std_val = float(group_data.std().mean())    # How spread out values are
        
        # Get all values and remove zeros for better analysis
        all_values = group_data.values.flatten()
        non_zero_values = all_values[all_values != 0]
        
        # Set smart limits based on data type
        if group == 0:  # PPI_0_* - Inter-packet timing data
            # For timing, use 98th percentile to avoid extreme outliers
            p98 = np.percentile(all_values, 98) if len(all_values) > 0 else max_val
            upper_threshold_stats = mean_val + 3 * std_val
            upper_threshold = min(p98, upper_threshold_stats) if p98 > 0 else upper_threshold_stats
            
            # Don't be too restrictive with the upper limit
            upper_threshold = max(upper_threshold, mean_val + 2.5 * std_val)
            
            adaptive_thresholds = {
                'lower_threshold': 0.0,                   # No negative timing allowed
                'upper_threshold': upper_threshold,       # Reasonable max timing
                'noise_std': std_val * 0.005              # Small amount of noise (0.5% of std)
            }
            
        elif group == 1:  # PPI_1_* - Packet direction (should be -1, 0, or +1)
            # For direction features, check what values actually exist
            unique_vals = np.unique(all_values)
            binary_tolerance = 0.05  # How close to exact binary values to accept
            
            adaptive_thresholds = {
                'binary_tolerance': binary_tolerance,
                'symmetry_break_chance': 0.05,              # 5% chance to break symmetry for variety
                'expected_binary_values': [-1.0, 1.0]       # Expected direction values
            }
            
        elif group == 2:  # PPI_2_* - Packet size data
            # For packet size, use 95th percentile (more conservative for timing)
            p95 = np.percentile(all_values, 95) if len(all_values) > 0 else max_val
            upper_threshold_stats = mean_val + 2.5 * std_val
            upper_threshold = min(p95, upper_threshold_stats) if p95 > 0 else upper_threshold_stats
            
            # Ensure reasonable minimum threshold
            upper_threshold = max(upper_threshold, mean_val + 2.0 * std_val)
            
            adaptive_thresholds = {
                'lower_threshold': 0.0,                     # No negative packet sizes
                'upper_threshold': upper_threshold,         # Reasonable max packet size
                'noise_std': min(std_val * 0.05, 10.0)      # Small noise: 5% of std or max 10ms
            }
        
        # Store all the statistics for this group
        analysis['ppi_groups'][group] = {
            'columns': group_cols,
            'value_distributions': {},
            'stats': {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'p25': float(np.percentile(all_values, 25)) if len(all_values) > 0 else min_val,    # 25th percentile
                'p50': float(np.percentile(all_values, 50)) if len(all_values) > 0 else mean_val,   # 50th percentile (median)
                'p75': float(np.percentile(all_values, 75)) if len(all_values) > 0 else max_val,    # 75th percentile
                'p90': float(np.percentile(all_values, 90)) if len(all_values) > 0 else max_val,    # 90th percentile
                'p95': float(np.percentile(all_values, 95)) if len(all_values) > 0 else max_val,    # 95th percentile
                'zero_ratio': float(np.sum(all_values == 0) / len(all_values)) if len(all_values) > 0 else 0.0,  # Percentage of zeros
                'non_zero_count': len(non_zero_values)     # Count of non-zero values
            }
        }
        
        # Store adaptive thresholds
        analysis['adaptive_thresholds'][group] = adaptive_thresholds
        
        # Analyze value distributions for each column in this group
        for col in group_cols:
            values = data[col].dropna()           # Remove missing values
            unique_values = values.unique()       # Get all different values
            value_counts = values.value_counts()  # Count how often each value appears
            
            # Store distribution information for this column
            analysis['ppi_groups'][group]['value_distributions'][col] = {
                'unique_values': unique_values.tolist(),
                'value_counts': value_counts.to_dict(),
                'probabilities': (value_counts / len(values)).to_dict()     # Convert counts to probabilities
            }
    
    # Show analysis results in the log
    log_message("Data analysis results:")
    for group in range(3):
        stats = analysis['ppi_groups'][group]['stats']
        thresholds = analysis['adaptive_thresholds'][group]
        
        group_names = ["Inter-packet timing", "Packet direction", "Packet size"]
        log_message(f"  {group_names[group]} (PPI_{group}): {stats['min']:.1f} to {stats['max']:.1f}")
        log_message(f"    Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        log_message(f"    Zero ratio: {stats['zero_ratio']:.1%}, P90: {stats['p90']:.1f}, P95: {stats['p95']:.1f}")
        
        if group == 0:
            log_message(f"    Smart upper limit: {thresholds['upper_threshold']:.1f} (vs fixed 10,000)")
        elif group == 1:
            log_message(f"    Binary tolerance: {thresholds['binary_tolerance']:.1f}")
        elif group == 2:
            log_message(f"    Smart upper limit: {thresholds['upper_threshold']:.1f} (vs fixed 1,500)")
    
    return analysis

def sample_from_distribution_precision_guided(col_analysis: dict, group: int = 0, model_feedback: dict = None) -> float:
    """
    Generate a single realistic value based on the original data patterns.
    Uses domain knowledge about network traffic to make smart choices.
    """
    probabilities = col_analysis['probabilities']   # How often each value appears in original data
    values = list(probabilities.keys())             # All possible values from original data
    probs = list(probabilities.values())            # Probability of each value
    
    if not values or not probs:
        return 0.0
    
    # Convert to numpy array and handle edge cases
    probs_array = np.array(probs)
    if probs_array.sum() == 0:
        probs_array = np.ones(len(probs)) / len(probs)      # Use uniform distribution if all zeros
    
    # Apply network traffic domain knowledge to improve sampling
    if group == 0:  # Inter-packet timing 
        enhanced_probs = probs_array.copy()     # For timing data, prefer realistic inter-packet times
        
        # Boost realistic inter-packet times (1ms to 100ms is typical for network traffic)
        for i, val in enumerate(values):
            # Realistic timing range
            if 0.001 <= val <= 100.0:           
                enhanced_probs[i] *= 2.0
            
            # Somewhat realistic
            elif 0.0001 <= val <= 0.001 or 100.0 < val <= 1000.0:  
                enhanced_probs[i] *= 1.3
            
            # Unrealistic long delays
            elif val > 10000.0:                 
                enhanced_probs[i] *= 0.3
        
        # Boost high-frequency patterns that represent real traffic
        high_freq_mask = probs_array > np.percentile(probs_array, 75)
        enhanced_probs[high_freq_mask] *= 1.5
        
    elif group == 1:  # Packet direction
        enhanced_probs = probs_array.copy()     # For direction, strongly prefer exact binary values
        
        for i, val in enumerate(values):
            # Exact -1.0 (incoming packets)
            if abs(val - (-1.0)) < 0.001:       
                enhanced_probs[i] *= 10.0
            
            # Exact +1.0 (outgoing packets) 
            elif abs(val - 1.0) < 0.001:        
                enhanced_probs[i] *= 10.0
            
            # Exact 0.0 (end of flow, padding)
            elif abs(val) < 0.001:              
                enhanced_probs[i] *= 5.0
            
            # Non-binary values (should be rare in good traffic data)
            else:                               
                enhanced_probs[i] *= 0.1
        
    elif group == 2:  # Packet size - prefer realistic network packet sizes
        enhanced_probs = probs_array.copy()     # For packet size, prefer realistic network packet sizes
        
        for i, val in enumerate(values):
            # Standard Ethernet packet sizes
            if 20 <= val <= 1500:               
                enhanced_probs[i] *= 2.5
            
            # Jumbo frames
            elif 1500 < val <= 9000:            
                enhanced_probs[i] *= 1.5
            
            # Very small packets (headers only)
            elif 1 <= val < 20:                 
                enhanced_probs[i] *= 1.2
            
            # Unrealistically large packets
            elif val > 9000:                    
                enhanced_probs[i] *= 0.2
            
            # Zero-size packets (might be valid in some contexts)
            elif val == 0:                      
                enhanced_probs[i] *= 0.8
        
        # Boost common packet sizes that appear frequently in original data
        high_freq_mask = probs_array > np.percentile(probs_array, 70)
        enhanced_probs[high_freq_mask] *= 1.4
    
    # Apply AI model feedback if available to further improve quality
    if model_feedback and 'preferred_patterns' in model_feedback:
        preferred_values = model_feedback['preferred_patterns'].get(group, {})
        for i, val in enumerate(values):
            if val in preferred_values:
                enhanced_probs[i] *= 1.5  # Moderate boost for model-preferred patterns
    
    # Make sure probabilities sum to 1
    enhanced_probs = enhanced_probs / enhanced_probs.sum()
    
    # Select a value based on the enhanced probabilities
    sampled_value = np.random.choice(values, p=enhanced_probs)
    
    # Clean up the value based on data type (post-processing)
    if group == 0:  # Inter-packet timing
        if sampled_value < 0:
            sampled_value = abs(sampled_value)                          # Make positive (timing can't be negative)
        # Add small timing variation for realism (network timing has natural jitter)
        if sampled_value > 0 and np.random.random() < 0.1:              # 10% chance for micro-jitter
            jitter = sampled_value * np.random.uniform(-0.02, 0.02)     # 2% timing jitter
            sampled_value = max(0.0001, sampled_value + jitter)
            
    elif group == 1:  # Packet direction - force to exact binary values
        # Network direction data should be exactly -1, 0, or +1
        if abs(sampled_value - (-1.0)) < 0.2:
            sampled_value = -1.0  # Incoming packets
        elif abs(sampled_value - 1.0) < 0.2:
            sampled_value = 1.0   # Outgoing packets
        else:
            sampled_value = 0.0   # End of flow (padding)
            
    elif group == 2:  # Packet size
        if sampled_value < 0:
            sampled_value = abs(sampled_value)  # Make positive (packet size can't be negative)
        sampled_value = float(int(round(sampled_value)))  # Round to whole bytes (packets are integer bytes)
        if 0 < sampled_value < 20:
            sampled_value = 20.0  # Minimum realistic packet size (Ethernet + IP headers)
    
    return sampled_value

# =============================================================================
# REFINEMENT FUNCTIONS - Improving the generated data quality
# =============================================================================

def apply_smart_refinement(sample: dict, adaptive_thresholds: dict, perplexity: float = None) -> dict:
    """
    Apply SMART refinement optimized for network traffic data.
    Only fixes clear violations without destroying valid patterns.
    """
    refined_sample = sample.copy()
    refinement_applied = False
    
    # Apply refinement rules for each PPI group - but only when necessary
    for group in range(3):
        group_cols = [f"PPI_{group}_{i}" for i in range(30)]
        thresholds = adaptive_thresholds[group]
        
        for col in group_cols:
            if col not in refined_sample:
                continue
                
            col_value = refined_sample[col]
            
            if group == 0:  # PPI_0_* - Inter-packet time: only fix clear violations
                if col_value < 0:  # Negative timing is always wrong
                    refined_sample[col] = abs(col_value)  # Make positive
                    refinement_applied = True
                elif col_value > thresholds['upper_threshold'] * 3:  # Only fix extreme outliers (3x threshold)
                    # Cap at reasonable maximum inter-packet time (10 seconds)
                    refined_sample[col] = min(10000.0, col_value * 0.5)
                    refinement_applied = True
                    
            elif group == 1:  # PPI_1_* - Packet direction: ensure proper binary values
                # For packet direction, we want exactly -1.0, 0.0, or +1.0
                tolerance = thresholds['binary_tolerance']
                
                # Check if it's close to a valid binary value
                is_near_minus_one = abs(col_value - (-1.0)) <= tolerance
                is_near_plus_one = abs(col_value - 1.0) <= tolerance
                is_near_zero = abs(col_value) <= tolerance
                
                if not (is_near_minus_one or is_near_plus_one or is_near_zero):
                    # Value is not close to any valid binary value - fix it
                    distances = [abs(col_value + 1), abs(col_value - 1), abs(col_value)]
                    closest_idx = np.argmin(distances)
                    refined_sample[col] = [-1.0, 1.0, 0.0][closest_idx]
                    refinement_applied = True
                elif is_near_minus_one:
                    refined_sample[col] = -1.0  # Snap to exact value
                    refinement_applied = True
                elif is_near_plus_one:
                    refined_sample[col] = 1.0   # Snap to exact value
                    refinement_applied = True
                elif is_near_zero:
                    refined_sample[col] = 0.0   # Snap to exact value
                    refinement_applied = True
                    
            elif group == 2:  # PPI_2_* - Packet size: ensure realistic values
                if col_value < 0:  # Negative packet size is wrong
                    refined_sample[col] = abs(col_value)
                    refinement_applied = True
                elif col_value > thresholds['upper_threshold'] * 2:  # Only fix extreme packet sizes
                    # Cap at reasonable maximum packet size (jumbo frame: 9000 bytes)
                    refined_sample[col] = min(9000.0, col_value * 0.6)
                    refinement_applied = True
                elif 0 < col_value < 20:  # Unrealistically small packet
                    # Minimum realistic packet size (Ethernet header + IP header)
                    refined_sample[col] = 20.0
                    refinement_applied = True
    
    return refined_sample, refinement_applied

# =============================================================================
# GENERATION FUNCTIONS - Creating synthetic data
# =============================================================================

def generate_data_aware_samples(analysis: dict, n_samples: int) -> list:
    """Generate synthetic samples using smart sampling based on original data patterns"""
    log_message(f"Generating {n_samples} precision-guided samples...")
    
    samples = []  # List to store all generated samples
    
    # Extract patterns from analysis to guide generation
    model_feedback = extract_model_feedback_patterns(analysis)
    log_message("Extracted model feedback patterns for enhanced sampling")
    
    # Generate each sample one by one
    for i in range(n_samples):
        # Show progress every 200 samples
        if i % 200 == 0:  
            log_message(f"  Generating sample {i+1}/{n_samples}...")
            
        sample = {}  # Dictionary to store one sample's data
        
        # Generate values for all PPI groups (0, 1, 2)
        for group in range(3):
            group_info = analysis['ppi_groups'][group]
            
            # Generate values for all 30 columns in this group
            for col in group_info['columns']:
                col_dist = group_info['value_distributions'][col]
                value = sample_from_distribution_precision_guided(col_dist, group, model_feedback)
                sample[col] = value
        
        # Pick a random application label from the original data
        app_labels = analysis['app_labels']
        sample['APP'] = np.random.choice(app_labels)
        
        samples.append(sample)      # Add completed sample to list
    
    return samples

def extract_model_feedback_patterns(analysis: dict) -> dict:
    """Find the most important patterns in the data to guide generation"""
    model_feedback = {'preferred_patterns': {}}
    
    for group in range(3):
        group_patterns = {}
        group_info = analysis['ppi_groups'][group]
        
        # Find high-frequency, realistic patterns for each group based on network traffic knowledge
        for col, col_info in group_info['value_distributions'].items():
            probabilities = col_info['probabilities']
            
            # Identify preferred values based on domain knowledge
            for value, prob in probabilities.items():
                if group == 0:  # Inter-packet timing
                    # Prefer realistic timing patterns
                    if prob > 0.05 and 0.0001 <= value <= 1000.0:   # Realistic timing range
                        group_patterns[value] = prob * 2.0          # Boost realistic timings
                    elif prob > 0.08:                               # Very common patterns (even if outside ideal range)
                        group_patterns[value] = prob * 1.5
                        
                elif group == 1:  # Packet direction
                    # Strongly prefer exact binary values for direction
                    if value in [-1.0, 0.0, 1.0] and prob > 0.02:
                        group_patterns[value] = prob * 5.0          # Very strong boost for clean binary
                    elif abs(value - (-1.0)) < 0.01 or abs(value - 1.0) < 0.01:
                        group_patterns[value] = prob * 3.0          # Strong boost for near-binary
                        
                elif group == 2:  # Packet size
                    # Prefer realistic packet sizes common in network traffic
                    if 20 <= value <= 1500 and prob > 0.04:         # Standard Ethernet frames
                        group_patterns[value] = prob * 3.0
                    elif 1500 < value <= 9000 and prob > 0.02:      # Jumbo frames
                        group_patterns[value] = prob * 2.0
                    elif prob > 0.08:                               # Very common sizes (even if unusual)
                        group_patterns[value] = prob * 1.5
        
        model_feedback['preferred_patterns'][group] = group_patterns
    
    # Log pattern statistics for debugging
    for group in range(3):
        patterns = model_feedback['preferred_patterns'][group]
        group_names = ["Inter-packet timing", "Packet direction", "Packet size"]
        log_message(f"   {group_names[group]} (PPI_{group}): {len(patterns)} preferred patterns identified")
    
    return model_feedback

def apply_model_perplexity_guidance(samples: list, model_path: str, analysis: dict) -> list:
    """
    Apply smart model-guided refinement that only improves truly problematic samples.
    Uses quality-preserving validation to ensure refinements actually help.
    """
    if not os.path.exists(model_path):
        log_message(f"Model not found: {model_path}. Skipping model guidance.")
        return samples  # Return original samples without aggressive refinement
    
    log_message("Applying smart model-guided refinement...")
    
    try:
        from network_traffic_generator.model import NetworkTrafficModel
        
        # Load the model
        model = NetworkTrafficModel.load_model(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        log_message(f"   Model loaded on {device}")
        
        # Get adaptive thresholds from analysis
        adaptive_thresholds = analysis['adaptive_thresholds']
        
        # Calculate perplexity for all samples
        perplexities = []
        encoding_failures = 0
        
        with torch.no_grad():
            for i, sample in enumerate(samples):
                if i % 200 == 0:
                    log_message(f"   Computing perplexity {i+1}/{len(samples)}...")
                
                try:
                    # Encode sample and calculate perplexity
                    sample_dict = {k: v for k, v in sample.items()}
                    tokens = model.encode_network_data(sample_dict)
                    
                    if len(tokens) > 0:
                        input_ids = torch.tensor([tokens], device=device)
                        attention_mask = torch.ones_like(input_ids)
                        
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                        loss = outputs['loss'].item()
                        perplexity = np.exp(loss)
                    else:
                        perplexity = 1000.0         # Very high perplexity for encoding failures
                        encoding_failures += 1
                    
                    perplexities.append(perplexity)
                    
                except Exception as e:
                    perplexities.append(1000.0)     # Very high perplexity for errors
                    encoding_failures += 1
        
        # More sophisticated sample selection for refinement
        perplexities = np.array(perplexities)
        
        # Use percentile-based thresholds instead of fixed values
        p75_perplexity = np.percentile(perplexities, 75)
        p90_perplexity = np.percentile(perplexities, 90)
        p95_perplexity = np.percentile(perplexities, 95)
        
        # Only refine samples in the top 10% that are truly problematic
        refinement_threshold = max(p90_perplexity, PERPLEXITY_THRESHOLD)
        
        # Be even more selective - only refine top 5% if perplexity distribution is reasonable
        median_perplexity = np.median(perplexities)
        if median_perplexity < 50:      # If most samples are reasonable
            refinement_threshold = p95_perplexity
            max_refinement_rate = 0.05  # Only refine 5% of samples
        else:
            max_refinement_rate = 0.10  # At most 10% of samples
        
        # Select samples for refinement
        refinement_candidates = np.where(perplexities > refinement_threshold)[0]
        
        # Limit the number of samples to refine
        max_to_refine = int(len(samples) * max_refinement_rate)
        if len(refinement_candidates) > max_to_refine:
            # Select the worst samples
            worst_indices = np.argsort(perplexities)[-max_to_refine:]
            refinement_candidates = worst_indices
        
        log_message(f"   Perplexity statistics:")
        log_message(f"     Median: {median_perplexity:.2f}, P75: {p75_perplexity:.2f}, P90: {p90_perplexity:.2f}")
        log_message(f"     Refinement threshold: {refinement_threshold:.2f}")
        log_message(f"     Encoding failures: {encoding_failures}")
        log_message(f"   Selected {len(refinement_candidates)}/{len(samples)} samples for potential refinement")
        
        # Apply smart refinement with quality validation
        refined_samples = []
        successful_refinements = 0
        
        for i, sample in enumerate(samples):
            if i in refinement_candidates:
                # Try refinement
                refined_sample, was_refined = apply_smart_refinement(
                    sample, adaptive_thresholds, perplexities[i]
                )
                
                if was_refined:
                    # Validate that refinement actually helps
                    try:
                        # Quick validation: check if refined sample can be encoded properly
                        refined_tokens = model.encode_network_data(refined_sample)
                        
                        if len(refined_tokens) > 0:
                            # Calculate new perplexity to see if it improved
                            input_ids = torch.tensor([refined_tokens], device=device)
                            attention_mask = torch.ones_like(input_ids)
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                            new_perplexity = np.exp(outputs['loss'].item())
                            
                            # Only keep refinement if it actually improves perplexity
                            if new_perplexity < perplexities[i] * 0.95:  # At least 5% improvement
                                refined_samples.append(refined_sample)
                                successful_refinements += 1
                            else:
                                refined_samples.append(sample)  # Keep original
                        else:
                            refined_samples.append(sample)  # Keep original if encoding fails
                    except:
                        refined_samples.append(sample)  # Keep original if validation fails
                else:
                    refined_samples.append(sample)  # No refinement needed
            else:
                refined_samples.append(sample.copy())  # No refinement needed
        
        log_message(f"   Applied refinement to {len(refinement_candidates)} candidates")
        log_message(f"   Successfully improved {successful_refinements} samples")
        log_message(f"   Refinement success rate: {100*successful_refinements/max(1,len(refinement_candidates)):.1f}%")
        
        return refined_samples
        
    except Exception as e:
        log_message(f"Model guidance failed: {e}")
        traceback.print_exc()
        # Return original samples instead of applying aggressive refinement
        log_message("   Returning original samples without refinement")
        return samples

# =============================================================================
# DUPLICATE CHECKING - Ensuring uniqueness from original data
# =============================================================================

def check_for_duplicates(generated_samples: list, original_data_path: str) -> dict:
    """Check if any generated samples are identical to original data samples"""
    log_message("Checking for duplicates between generated and original data...")
    
    try:
        # Load original data for comparison
        original_data = pd.read_csv(original_data_path)
        log_message(f"   Loaded {len(original_data)} original samples for comparison")
        
        # Convert generated samples to DataFrame
        generated_df = pd.DataFrame(generated_samples)
        
        # Get columns to compare (all PPI columns plus APP label)
        ppi_cols = [f"PPI_{g}_{i}" for g in range(3) for i in range(30)]
        columns_to_compare = ppi_cols + ['APP']
        
        # Round values to handle floating point precision issues
        original_rounded = original_data[columns_to_compare].round(6)
        generated_rounded = generated_df[columns_to_compare].round(6)
        
        # Convert rows to strings for exact comparison
        original_strings = original_rounded.apply(lambda row: '|'.join(row.astype(str)), axis=1)
        generated_strings = generated_rounded.apply(lambda row: '|'.join(row.astype(str)), axis=1)
        
        # Find duplicates
        original_set = set(original_strings)
        duplicates = []
        duplicate_indices = []
        
        for i, gen_string in enumerate(generated_strings):
            if gen_string in original_set:
                duplicates.append(generated_samples[i])
                duplicate_indices.append(i)
        
        # Calculate statistics
        num_duplicates = len(duplicates)
        duplicate_percentage = (num_duplicates / len(generated_samples)) * 100
        
        log_message(f"   Duplicate check results:")
        log_message(f"     Generated samples: {len(generated_samples)}")
        log_message(f"     Original samples: {len(original_data)}")
        log_message(f"     Exact duplicates found: {num_duplicates}")
        log_message(f"     Duplicate percentage: {duplicate_percentage:.2f}%")
        
        if num_duplicates > 0:
            log_message(f"   WARNING: {num_duplicates} generated samples are identical to original data")
        else:
            log_message(f"   Good: No duplicates found - all generated samples are unique")
        
        return {
            'num_duplicates': num_duplicates,
            'duplicate_percentage': duplicate_percentage,
            'duplicate_indices': duplicate_indices,
            'total_generated': len(generated_samples),
            'total_original': len(original_data)
        }
        
    except Exception as e:
        log_message(f"   Error during duplicate check: {e}")
        return {'num_duplicates': -1, 'duplicate_percentage': -1, 'error': str(e)}

def save_samples(samples: list, analysis: dict, mode: str, model_guidance_used: bool):
    """Save generated samples and analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Ensure proper column order
    ppi_cols = [f"PPI_{g}_{i}" for g in range(3) for i in range(30)]
    df = df[ppi_cols + ['APP']]
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"synthetic_{mode}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save analysis
    analysis_path = os.path.join(OUTPUT_DIR, f"synthetic_{mode}_{timestamp}_analysis.json")
    with open(analysis_path, 'w') as f:
        json_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                json_analysis[key] = {k: v for k, v in value.items()}
            else:
                json_analysis[key] = value
        json.dump(json_analysis, f, indent=2, default=str)
    
    # Check for duplicates with original data
    duplicate_stats = check_for_duplicates(samples, ORIGINAL_DATA_PATH)
    
    # Save configuration
    config_path = os.path.join(OUTPUT_DIR, f"synthetic_{mode}_{timestamp}_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"SIMPLIFIED NETWORK TRAFFIC GENERATION - {mode.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generation Mode: {mode}\n")
        f.write(f"Original Data: {ORIGINAL_DATA_PATH}\n")
        f.write(f"Samples Generated: {len(df)}\n")
        f.write(f"Model Guidance: {model_guidance_used}\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        if model_guidance_used:
            f.write(f"Perplexity Threshold: {PERPLEXITY_THRESHOLD}\n")
        f.write(f"Generated Shape: {df.shape}\n")
        
        # Add adaptive thresholds information (only if available)
        if 'adaptive_thresholds' in analysis:
            f.write(f"\nADAPTIVE ALGORITHM 1 THRESHOLDS:\n")
            for group in range(3):
                thresholds = analysis['adaptive_thresholds'][group]
                if group == 0:
                    f.write(f"PPI_0: upper_threshold = {thresholds['upper_threshold']:.1f} (vs fixed 10,000)\n")
                    f.write(f"       noise_std = {thresholds['noise_std']:.3f}\n")
                elif group == 1:
                    f.write(f"PPI_1: binary_tolerance = ±{thresholds['binary_tolerance']:.1f}\n")
                    f.write(f"       symmetry_break_chance = {thresholds['symmetry_break_chance']:.1%}\n")
                elif group == 2:
                    f.write(f"PPI_2: upper_threshold = {thresholds['upper_threshold']:.1f} (vs fixed 1,500)\n")
                    f.write(f"       noise_std = {thresholds['noise_std']:.3f}\n")
        
        # Add duplicate check results
        f.write(f"\nDUPLICATE CHECK RESULTS:\n")
        f.write(f"Exact duplicates found: {duplicate_stats.get('num_duplicates', 'Error')}\n")
        f.write(f"Duplicate percentage: {duplicate_stats.get('duplicate_percentage', 'Error'):.2f}%\n")
        f.write(f"Original samples compared: {duplicate_stats.get('total_original', 'Error')}\n")
        
        # Add distribution summary
        f.write("\nPPI GROUP SUMMARIES:\n")
        for group in range(3):
            group_cols = [f"PPI_{group}_{i}" for i in range(30)]
            group_data = df[group_cols]
            unique_vals = len(pd.concat([group_data[col] for col in group_cols]).unique())
            non_zero_count = (group_data != 0).sum().sum()
            total_count = len(group_data) * len(group_cols)
            f.write(f"PPI Group {group}: {group_data.min().min():.1f} to {group_data.max().max():.1f}\n")
            f.write(f"  Unique values: {unique_vals}, Non-zero: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)\n")
    
    log_message(f"\nSaved results:")
    log_message(f"  Data: {csv_path}")
    log_message(f"  Analysis: {analysis_path}")
    log_message(f"  Config: {config_path}")
    log_message(f"  Shape: {df.shape}")
    
    return csv_path

# =============================================================================
# VALIDATION FUNCTION - Ensuring quality of generated samples
# =============================================================================

def validate_sample_quality(samples: list, mode: str) -> dict:
    """
    Check the quality of generated samples and give scores for each data type.
    Returns detailed quality metrics.
    """
    log_message(f"Validating sample quality for {mode} mode...")
    
    if not samples:
        return {'error': 'No samples to validate'}
    
    # Convert samples to DataFrame for analysis
    df = pd.DataFrame(samples)
    
    quality_metrics = {
        'mode': mode,
        'total_samples': len(samples),
        'ppi_group_quality': {}
    }
    
    # Check quality for each PPI group separately
    for group in range(3):
        group_cols = [f"PPI_{group}_{i}" for i in range(30)]
        group_data = df[group_cols]
        
        if group == 0:  # Inter-packet timing quality
            negative_count = (group_data < 0).sum().sum()
            zero_count = (group_data == 0).sum().sum()
            realistic_timing = ((group_data >= 0.0001) & (group_data <= 1000.0)).sum().sum()
            extreme_outliers = (group_data > 10000.0).sum().sum()
            
            # Calculate quality score (100 is perfect)
            quality_score = max(0, 100 - (negative_count * 10) - (extreme_outliers * 5))
            
            quality_metrics['ppi_group_quality'][0] = {
                'type': 'Inter-packet timing',
                'negative_values': int(negative_count),
                'zero_values': int(zero_count),
                'realistic_timing_values': int(realistic_timing),
                'extreme_outliers': int(extreme_outliers),
                'quality_score': quality_score
            }
            
        elif group == 1:  # Packet direction quality
            exact_minus_one = (group_data == -1.0).sum().sum()
            exact_plus_one = (group_data == 1.0).sum().sum() 
            exact_zero = (group_data == 0.0).sum().sum()
            near_binary = ((np.abs(group_data + 1) < 0.1) | 
                          (np.abs(group_data - 1) < 0.1) | 
                          (np.abs(group_data) < 0.1)).sum().sum()
            total_values = group_data.size
            
            # Calculate ratio of exact binary values
            exact_binary_ratio = (exact_minus_one + exact_plus_one + exact_zero) / total_values
            
            quality_metrics['ppi_group_quality'][1] = {
                'type': 'Packet direction (binary)',                # Use 'binary' to indicate exact values
                'exact_minus_one': int(exact_minus_one),            # Exact -1.0 (incoming packets)
                'exact_plus_one': int(exact_plus_one),              # Exact +1.0 (outgoing packets)
                'exact_zero': int(exact_zero),                      # Exact 0.0 (neutral/unknown direction)
                'exact_binary_ratio': float(exact_binary_ratio),    # Ratio of exact binary values
                'near_binary_values': int(near_binary),             # Values close to binary (-1, 0, +1)
                'quality_score': exact_binary_ratio * 100           # Perfect binary gets 100%
            }
            
        elif group == 2:  # Packet size quality
            negative_count = (group_data < 0).sum().sum()
            zero_count = (group_data == 0).sum().sum()
            realistic_packets = ((group_data >= 20) & (group_data <= 9000)).sum().sum()
            tiny_packets = ((group_data > 0) & (group_data < 20)).sum().sum()
            huge_packets = (group_data > 9000).sum().sum()
            integer_packets = (group_data == group_data.round()).sum().sum()
            
            # Calculate quality score
            quality_score = max(0, 100 - (negative_count * 10) - (tiny_packets * 2) - (huge_packets * 5))
            
            quality_metrics['ppi_group_quality'][2] = {
                'type': 'Packet size',                              # Use 'size' to indicate packet sizes
                'negative_values': int(negative_count),             # Negative packet sizes
                'zero_values': int(zero_count),                     # Zero-size packets
                'realistic_packet_sizes': int(realistic_packets),   # Realistic packet sizes (20-9000 bytes)
                'tiny_packets': int(tiny_packets),                  # Tiny packets (1-19 bytes)
                'huge_packets': int(huge_packets),                  # Unrealistically large packets (>9000 bytes)
                'integer_values': int(integer_packets),             # Integer packet sizes (whole bytes)
                'quality_score': quality_score                      # Quality score based on realistic sizes
            }
    
    # Calculate overall quality score (average of all groups)
    group_scores = [quality_metrics['ppi_group_quality'][g]['quality_score'] for g in range(3)]
    quality_metrics['overall_quality_score'] = sum(group_scores) / 3
    
    # Show results in log
    log_message(f"   Quality validation results for {mode}:")
    for group in range(3):
        metrics = quality_metrics['ppi_group_quality'][group]
        log_message(f"     {metrics['type']}: {metrics['quality_score']:.1f}/100")
        
        if group == 0:  # Inter-packet timing
            log_message(f"       Negative values: {metrics['negative_values']}")
            log_message(f"       Extreme outliers: {metrics['extreme_outliers']}")
        elif group == 1:  # Packet direction
            log_message(f"       Exact binary ratio: {metrics['exact_binary_ratio']:.1%}")
        elif group == 2:  # Packet size
            log_message(f"       Negative: {metrics['negative_values']}")
            log_message(f"       Tiny packets: {metrics['tiny_packets']}")
            log_message(f"       Huge packets: {metrics['huge_packets']}")
    
    log_message(f"   Overall quality score: {quality_metrics['overall_quality_score']:.1f}/100")
    
    return quality_metrics

# =============================================================================
# MAIN GENERATING SCRIPT
# =============================================================================

def main():
    """    
    This is where everything happens! The main function:
    1. Sets up logging and creates output folders
    2. Loads your original network traffic data
    3. Analyzes the data patterns and characteristics
    4. Generates synthetic data using AI algorithms
    5. Validates the quality of generated data
    6. Saves results and creates detailed reports
    """
    global LOG_FILE
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(OUTPUT_DIR, f"generation_log_{timestamp}.txt")
    
    log_message("=" * 60)
    log_message(f"PRECISION-GUIDED NETWORK TRAFFIC GENERATION")
    log_message("=" * 60)
    log_message(f"Start at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Mode: {GENERATION_MODE}")
    log_message(f"Samples: {N_SAMPLES}")
    log_message(f"Model Guidance: {USE_MODEL_GUIDANCE}")
    
    if USE_MODEL_GUIDANCE:
        log_message(f"Perplexity Threshold: {PERPLEXITY_THRESHOLD}")
        log_message(f"Force Refinement: {FORCE_REFINEMENT_PERCENTAGE*100}%")
        
    log_message("")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    import time
    start_time = time.time()
    try:
        # Load original data
        log_message("Loading original data...")
        if not os.path.exists(ORIGINAL_DATA_PATH):
            log_message(f"Original data not found: {ORIGINAL_DATA_PATH}")
            return 1
        original_data = pd.read_csv(ORIGINAL_DATA_PATH)
        log_message(f"Loaded {len(original_data)} original samples")

        # Generate samples using precision-guided approach
        log_message("Using ENHANCED precision-guided sampling + smart model refinement...")
        
        # Step 1: Analyze data with adaptive thresholds
        analysis = analyze_data_distributions(original_data)
        
        # Step 2: Generate with precision-guided distribution sampling
        samples = generate_data_aware_samples(analysis, N_SAMPLES)
        
        # Step 3: Validate pre-refinement quality
        pre_refinement_quality = validate_sample_quality(samples, "precision_guided_pre_refinement")
        
        # Step 4: Apply smart model-guided refinement (if enabled)
        if USE_MODEL_GUIDANCE:
            samples = apply_model_perplexity_guidance(samples, MODEL_PATH, analysis)
            # Validate post-refinement quality
            quality_metrics = validate_sample_quality(samples, "precision_guided_post_refinement")
            # Compare pre vs post refinement
            improvement = quality_metrics['overall_quality_score'] - pre_refinement_quality['overall_quality_score']
            log_message(f"Model refinement impact: {improvement:+.1f} points")
        else:
            log_message("   Skipping model guidance (disabled in settings)")
            quality_metrics = pre_refinement_quality

        # Save results
        if samples:
            output_path = save_samples(samples, analysis, GENERATION_MODE, USE_MODEL_GUIDANCE)
            end_time = time.time()
            elapsed = end_time - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"Generation completed in: {hours}h {minutes}m {seconds}s"
            print(elapsed_str)
            log_message(elapsed_str)
            log_message("\nGeneration completed successfully!")
            log_message(f"Generated {len(samples)} samples using precision-guided approach")
            return 0
        else:
            log_message("ERROR: No samples generated! Check the log for error details.")
            return 1
    except Exception as e:
        log_message(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
