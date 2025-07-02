#!/usr/bin/env python3
"""
Tree Data Extraction Module

Extracts branch-and-bound tree information from solver log files (.out files).
Parses tree sections, bounds, and branching decisions to create structured tree data.

Author: Heinrich (Refined)
"""

import re
import os
import glob
from pathlib import Path


def extract_tree_sections(file_path, debug=False):
    """
    Parse the .out file to extract tree information and branching details for 2-way branching.
    Extracts the final LP value as the last EXACT pricing before "evaluate on edge:" markers.
    
    Args:
        file_path (str): Path to the solver log file
        debug (bool): Enable debug output
        
    Returns:
        list: List of tree section dictionaries with extracted data
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    # Find all sections enclosed by continuous '#' lines for tree information
    pattern = r'(#{81})\s*tree size=\s*(\d+)\s*lb=\s*(\d+(?:\.\d+)?)\s*ub=\s*(\d+(?:\.\d+)?)\s*#{81}(.*?)(?=#{81}|$)'
    sections = re.findall(pattern, content, re.DOTALL)
    
    if debug:
        print(f"Number of tree sections matched by regex: {len(sections)}")
    
    tree_sections = []
    
    for section_index, section in enumerate(sections):
        hashes, tree_size, initial_lb, ub, details = section
        
        if debug:
            print(f"\nProcessing section {section_index+1} (first 100 chars of details):")
            print(details[:100] + "..." if len(details) > 100 else details)
        
        # Extract idx and value
        idx_match = re.search(r'idx=\s*(\d+)', details)
        value_match = re.search(r'value=\s*(\d+(?:\.\d+)?)', details)
        
        # Check if we found idx - this is critical
        if not idx_match:
            if debug:
                print(f"WARNING: idx not found in section {section_index+1}")
            continue  # Skip this section if no idx found
        
        idx = int(idx_match.group(1))
        value = float(value_match.group(1)) if value_match else None
        
        if debug:
            print(f"Found idx={idx}")
        
        # Extract branching information for 2-way format
        branch_pattern = r'brc=\s*(\d+-\d+)\s*:\s*([+-]+)'
        branches_raw = re.findall(branch_pattern, details)
        
        # Convert to standard format with placeholder side="none"
        branches = [(edge, direction, "none") for edge, direction in branches_raw]
        
        # Initialize lb with the header value as fallback
        final_lb = float(initial_lb)
        
        # For 2-way branching: Check for "evaluate on edge:" marker
        edge_eval_marker = "evaluate on edge:"
        edge_eval_pos = details.find(edge_eval_marker)
        
        # Determine relevant text for finding exact pricing
        if edge_eval_pos >= 0:
            # Find text BEFORE the first edge evaluation
            relevant_text = details[:edge_eval_pos]
            if debug:
                print(f"Found 'evaluate on edge:' marker at position {edge_eval_pos}")
        else:
            # No marker found, use all text
            relevant_text = details
            if debug:
                print("No 'evaluate on edge:' marker found, using entire section")
        
        # Look for EXACT pricing cycles in the relevant text
        exact_pricing_pattern = r'\*{6}\s+Pricing Level:\s+EXACT\s+\*{6}.*?lpval=\s*(\d+\.\d+)'
        exact_pricing_matches = re.findall(exact_pricing_pattern, relevant_text, re.DOTALL)
        
        # If we found any EXACT pricing cycles, use the last one's LP value as the final lb
        if exact_pricing_matches:
            final_lb = float(exact_pricing_matches[-1])
            if debug:
                print(f"Updated LB from {initial_lb} to {final_lb} based on EXACT pricing")
                print(f"Found {len(exact_pricing_matches)} EXACT pricing cycles")
        else:
            if debug:
                print(f"No EXACT pricing cycles found, using initial LB: {initial_lb}")
        
        tree_section = {
            'tree_size': int(tree_size),
            'initial_lb': float(initial_lb),  # Keep the original lb for reference
            'lb': final_lb,  # Updated lb based on the final LP value
            'ub': float(ub),
            'idx': idx,
            'value': value,
            'branches': branches,
            'has_edge_eval_marker': edge_eval_pos >= 0,
            'details': details.strip()
        }
        
        tree_sections.append(tree_section)
    
    # Sort by idx to ensure correct order
    tree_sections.sort(key=lambda x: x['idx'])
    
    if debug:
        print_tree_debug_info(tree_sections)
        
        # Check for missing indexes
        idx_values = [section['idx'] for section in tree_sections]
        if idx_values:  # Only check if we have any data
            expected_idx = list(range(max(idx_values) + 1))
            missing_idx = [idx for idx in expected_idx if idx not in idx_values]
            if missing_idx:
                print("WARNING: Missing idx values:", missing_idx)
        
    return tree_sections


def print_tree_debug_info(tree_sections):
    """Print debug information about the parsed tree data."""
    print("\n" + "="*80)
    print("DEBUG: 2-Way Branch-and-Bound Tree Data Summary")
    print("="*80)
    
    print(f"Number of tree sections found: {len(tree_sections)}")
    
    for i, section in enumerate(tree_sections):
        print(f"\nTree Section {i+1} (idx={section['idx']}):")
        print(f"  Tree Size: {section['tree_size']}")
        print(f"  Has Edge Evaluation Marker: {section.get('has_edge_eval_marker', False)}")
        print(f"  Initial LB (from header): {section['initial_lb']}")
        print(f"  Final LB (from EXACT pricing): {section['lb']}")
        print(f"  UB: {section['ub']}")
        print(f"  Gap: {((section['ub'] - section['lb']) / section['ub'] * 100):.4f}%")
        print(f"  Value: {section['value']}")
        
        print(f"  Branches ({len(section['branches'])}):")
        for j, branch in enumerate(section['branches']):
            edge, direction, side = branch
            print(f"    {j+1}. Edge {edge}: {direction}")
        
        print("-"*40)
    
    print("\n" + "="*80 + "\n")


def format_tree_report(tree_sections, instance_name):
    """Generate formatted tree information output."""
    output_lines = []
    
    output_lines.append("="*80)
    output_lines.append(f"Tree Information Details for {instance_name}")
    output_lines.append("="*80)
    output_lines.append(f"Total tree sections found: {len(tree_sections)}")
    output_lines.append("")
    
    for i, section in enumerate(tree_sections):
        output_lines.append(f"Tree Section {i+1} (idx={section['idx']}):")
        output_lines.append(f"  Tree Size: {section['tree_size']}")
        output_lines.append(f"  Has Edge Evaluation Marker: {section.get('has_edge_eval_marker', False)}")
        output_lines.append(f"  Initial LB (from header): {section['initial_lb']:.6f}")
        output_lines.append(f"  Final LB (from EXACT pricing): {section['lb']:.6f}")
        output_lines.append(f"  UB: {section['ub']:.6f}")
        
        gap = ((section['ub'] - section['lb']) / section['ub'] * 100) if section['ub'] > 0 else 0
        output_lines.append(f"  Gap: {gap:.4f}%")
        output_lines.append(f"  Value: {section['value']}")
        
        output_lines.append(f"  Branches ({len(section['branches'])}):")
        for j, branch in enumerate(section['branches']):
            edge, direction, side = branch
            output_lines.append(f"    {j+1}. Edge {edge}: {direction}")
        
        output_lines.append("-"*40)
        output_lines.append("")
    
    return "\n".join(output_lines)


def get_instance_name(filepath):
    """Extract instance name from .out file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Read last 100 lines to find instance info
            lines = f.readlines()
            last_lines = lines[-100:] if len(lines) > 100 else lines
            content = ''.join(last_lines)
        
        # Extract instance name from the colored output line
        instance_match = re.search(r'<Instance:\s*([^>]+)>', content)
        if instance_match:
            return instance_match.group(1).strip()
        else:
            # Fallback: use filename without extension
            return os.path.splitext(os.path.basename(filepath))[0]
            
    except Exception:
        # Fallback: use filename without extension
        return os.path.splitext(os.path.basename(filepath))[0]


def create_output_filename(instance_name):
    """Generate tree output filename from instance name."""
    # Extract numbers from instance name (e.g., CVRP_120_100 -> 120_100)
    numbers = re.findall(r'\d+', instance_name)
    if len(numbers) >= 2:
        return f"tree_dg_{numbers[-2]}_{numbers[-1]}.txt"
    else:
        # Fallback if pattern doesn't match expected format
        clean_name = re.sub(r'[^\w\d_]', '_', instance_name)
        return f"tree_{clean_name}.txt"


def process_log_files(input_folder=".", output_dir="tree_details", debug_mode=False):
    """
    Process multiple .out files and extract tree data from each.
    
    Args:
        input_folder (str): Path to folder containing .out files
        output_dir (str): Directory to save tree detail files
        debug_mode (bool): Enable debug output
        
    Returns:
        int: Number of files processed successfully
    """
    # Find all .out files
    out_files = glob.glob(os.path.join(input_folder, "*.out"))
    
    if not out_files:
        print(f"No .out files found in {input_folder}")
        return 0
    
    print(f"Found {len(out_files)} .out files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Process each file
    processed_count = 0
    for filepath in out_files:
        print(f"Processing: {os.path.basename(filepath)}")
        
        # Extract instance name
        instance_name = get_instance_name(filepath)
        
        # Parse tree data
        tree_sections = extract_tree_sections(filepath, debug=debug_mode)
        
        if not tree_sections:
            print(f"  -> No tree data found in {os.path.basename(filepath)}")
            continue
        
        # Generate output filename
        output_filename = create_output_filename(instance_name)
        output_path = os.path.join(output_dir, output_filename)
        
        # Generate tree output
        tree_output = format_tree_report(tree_sections, instance_name)
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(tree_output)
            
            processed_count += 1
            print(f"  -> Created: {output_filename} ({len(tree_sections)} tree sections)")
            
        except Exception as e:
            print(f"  -> Error writing {output_filename}: {e}")
    
    return processed_count


def main():
    """Main function for command line usage."""
    # Get input folder
    input_folder = input("Enter the folder path containing .out files (or press Enter for current directory): ").strip()
    if not input_folder:
        input_folder = "."
    
    # Ask for debug mode
    debug_mode = input("Enable debug mode? (y/N): ").strip().lower() == 'y'
    
    # Process files
    processed_count = process_log_files(input_folder, debug_mode=debug_mode)
    
    print(f"\nProcessing complete! {processed_count} files processed successfully.")
    print(f"Tree detail files saved in: tree_details/")


if __name__ == "__main__":
    main()