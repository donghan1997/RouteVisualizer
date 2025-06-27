#!/usr/bin/env python3
import re
import os
import glob
from pathlib import Path
from collections import defaultdict, deque
import math

class TreeNode:
    def __init__(self, idx, data):
        self.idx = idx
        self.data = data
        self.parent = None
        self.children = []
        self.level = 0
        self.subtree_count = 1  # Includes self
        self.branch_prefix = []  # Common branching decisions leading to this node
        self.r_value = None  # Node-specific r value
        self.estimated_size = 1  # Node-specific estimated subtree size
    
    def add_child(self, child_node):
        child_node.parent = self
        child_node.level = self.level + 1
        self.children.append(child_node)
    
    def calculate_subtree_count(self):
        """Calculate the total number of nodes in this subtree (including self)"""
        if not self.children:
            self.subtree_count = 1
            return 1
        
        total = 1  # Count self
        for child in self.children:
            total += child.calculate_subtree_count()
        
        self.subtree_count = total
        return total

def parse_tree_data_file(file_path):
    """Parse tree data file and extract node information."""
    nodes = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}
    
    # Parse each tree section
    sections = re.findall(r'Tree Section \d+ \(idx=(\d+)\):(.*?)(?=Tree Section \d+|$)', content, re.DOTALL)
    
    for idx_str, section_content in sections:
        idx = int(idx_str)
        
        # Extract tree size
        tree_size_match = re.search(r'Tree Size: (\d+)', section_content)
        tree_size = int(tree_size_match.group(1)) if tree_size_match else 0
        
        # Extract other values
        has_marker_match = re.search(r'Has Edge Evaluation Marker: (True|False)', section_content)
        has_marker = has_marker_match.group(1) == 'True' if has_marker_match else False
        
        initial_lb_match = re.search(r'Initial LB \(from header\): ([\d.]+)', section_content)
        initial_lb = float(initial_lb_match.group(1)) if initial_lb_match else 0.0
        
        final_lb_match = re.search(r'Final LB \(from EXACT pricing\): ([\d.]+)', section_content)
        final_lb = float(final_lb_match.group(1)) if final_lb_match else 0.0
        
        ub_match = re.search(r'UB: ([\d.]+)', section_content)
        ub = float(ub_match.group(1)) if ub_match else 0.0
        
        gap_match = re.search(r'Gap: ([-\d.]+)%', section_content)
        gap = float(gap_match.group(1)) if gap_match else 0.0
        
        value_match = re.search(r'Value: ([\d.]+|None)', section_content)
        value = float(value_match.group(1)) if value_match and value_match.group(1) != 'None' else None
        
        # Extract branches
        branch_matches = re.findall(r'\d+\. Edge ([\d-]+): ([+-])', section_content)
        branches = [(edge, direction) for edge, direction in branch_matches]
        
        # Create node data
        node_data = {
            'tree_size': tree_size,
            'has_marker': has_marker,
            'initial_lb': initial_lb,
            'final_lb': final_lb,
            'ub': ub,
            'gap': gap,
            'value': value,
            'branches': branches
        }
        
        nodes[idx] = TreeNode(idx, node_data)
    
    return nodes

def find_branch_prefix(branches1, branches2):
    """Find common prefix of branching decisions between two branch lists."""
    common_prefix = []
    min_len = min(len(branches1), len(branches2))
    
    for i in range(min_len):
        if branches1[i] == branches2[i]:
            common_prefix.append(branches1[i])
        else:
            break
    
    return common_prefix

def build_tree_structure(nodes):
    """Build tree structure based on branching patterns and tree sizes."""
    if not nodes:
        return None
    
    # Sort nodes by idx to process in order
    sorted_nodes = sorted(nodes.values(), key=lambda x: x.idx)
    
    # Root is typically the first node (idx=0)
    root = sorted_nodes[0]
    root.level = 0
    root.branch_prefix = []
    
    # Build parent-child relationships
    for i in range(1, len(sorted_nodes)):
        current_node = sorted_nodes[i]
        current_branches = current_node.data['branches']
        
        # Find parent by looking for the most recent node with matching branch prefix
        best_parent = None
        max_common_prefix_len = -1
        
        for j in range(i-1, -1, -1):  # Look backwards
            candidate_parent = sorted_nodes[j]
            candidate_branches = candidate_parent.data['branches']
            
            # Check if candidate could be parent
            # Parent should have fewer branches (one less branching decision)
            if len(candidate_branches) < len(current_branches):
                common_prefix = find_branch_prefix(candidate_branches, current_branches)
                
                # If current node's branches start with candidate's branches
                if len(common_prefix) == len(candidate_branches):
                    if len(common_prefix) > max_common_prefix_len:
                        max_common_prefix_len = len(common_prefix)
                        best_parent = candidate_parent
        
        # If no direct parent found, try to find based on tree structure logic
        if best_parent is None and i > 0:
            # Fallback: use the previous node as parent if it makes sense
            prev_node = sorted_nodes[i-1]
            if len(prev_node.data['branches']) < len(current_branches):
                best_parent = prev_node
        
        if best_parent:
            best_parent.add_child(current_node)
            current_node.branch_prefix = current_node.data['branches'][:-1]  # All but last branch
        else:
            # If no parent found, make it a child of root
            if current_node != root:
                root.add_child(current_node)
                current_node.branch_prefix = []
    
    # Calculate subtree counts
    root.calculate_subtree_count()
    
    return root

def arrange_nodes_by_level(root):
    """Arrange nodes by tree level using BFS."""
    if not root:
        return []
    
    arranged_nodes = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            # Add children to queue for next level
            for child in sorted(node.children, key=lambda x: x.idx):
                queue.append(child)
        
        arranged_nodes.extend(current_level)
    
    return arranged_nodes

def safe_power_calculation(base, exponent, max_exponent=100):
    """Safely calculate base^exponent with overflow protection."""
    if exponent > max_exponent:
        return float('inf')  # Return infinity for very large values
    try:
        result = base ** exponent
        # Check if result is too large (Python's float limit is around 1.8e308)
        if result > 1e100:  # Use a more conservative limit
            return float('inf')
        return result
    except OverflowError:
        return float('inf')

def calculate_simplified_estimated_size_with_k(node, k_value):
    """
    SIMPLIFIED: Calculate estimated subtree size using only the basic formula 2^(G/(k*r))
    No special cases for gains > G
    """
    # If no children, estimated size is 1
    if not node.children:
        return 1
    
    # Calculate parent's G (optimality gap)
    G = node.data['ub'] - node.data['final_lb']
    
    # Handle edge cases for G
    if G <= 0:
        return 1
    
    # Get children's LB gains
    child_gains = []
    for child in node.children:
        child_gain = child.data['final_lb'] - child.data['initial_lb']
        child_gains.append(child_gain)
    
    # Filter positive gains
    positive_gains = [gain for gain in child_gains if gain > 0]
    
    # If no positive gains, return 1
    if not positive_gains:
        return 1
    
    # Calculate r value based on number of children with positive gains
    if len(positive_gains) == 1:
        r_value = positive_gains[0]
    elif len(positive_gains) == 2:
        r_value = math.sqrt(positive_gains[0] * positive_gains[1])
    else:
        # For multiple children, use geometric mean
        r_value = math.exp(sum(math.log(gain) for gain in positive_gains) / len(positive_gains))
    
    # Minimum r value to prevent division by zero
    min_r_value = max(G / 100, 0.001)
    r_value = max(r_value, min_r_value)
    
    # Apply the basic formula: 2^(G/(k*r))
    exponent = G / (k_value * r_value)
    return safe_power_calculation(2, exponent)

def calculate_good_estimate_ratio(arranged_nodes, k_value):
    """
    Calculate the good estimate ratio for a given k value.
    Good estimate: 0.5 < est/real < 2, excluding trivial nodes (real=1, est≈1)
    """
    valid_comparisons = 0
    good_estimates = 0
    
    for node in arranged_nodes:
        real_size = node.subtree_count
        
        # Skip trivial nodes (real=1, est≈1)
        if real_size == 1:
            continue
            
        # Calculate estimated size with this k value
        est_size = calculate_simplified_estimated_size_with_k(node, k_value)
        
        # Skip if estimated size is also trivial (≈1)
        if est_size <= 1.1:  # Allow small tolerance for floating point
            continue
            
        # This is a valid comparison
        valid_comparisons += 1
        
        # Check if it's a good estimate
        ratio = est_size / real_size
        if 0.5 <= ratio <= 2.0:
            good_estimates += 1
    
    # Return ratio (handle division by zero)
    if valid_comparisons == 0:
        return 0.0
    return good_estimates / valid_comparisons

def calculate_simplified_node_r_and_estimated_size(node, k_value=1.0):
    """
    SIMPLIFIED: Calculate node-specific r value and estimated subtree size.
    Uses only the basic formula 2^(G/(k*r)) - no special cases.
    """
    # If no children, estimated size is 1
    if not node.children:
        node.r_value = None
        node.estimated_size = 1
        return
    
    # Calculate parent's G (optimality gap)
    G = node.data['ub'] - node.data['final_lb']
    
    # Handle edge cases for G
    if G <= 0:
        node.r_value = None
        node.estimated_size = 1
        return
    
    # Get children's LB gains
    child_gains = []
    for child in node.children:
        child_gain = child.data['final_lb'] - child.data['initial_lb']
        child_gains.append(child_gain)
    
    # Filter positive gains
    positive_gains = [gain for gain in child_gains if gain > 0]
    
    # If no positive gains, return 1
    if not positive_gains:
        node.r_value = None
        node.estimated_size = 1
        return
    
    # Calculate r value based on number of children with positive gains
    if len(positive_gains) == 1:
        node.r_value = positive_gains[0]
    elif len(positive_gains) == 2:
        node.r_value = math.sqrt(positive_gains[0] * positive_gains[1])
    else:
        # For multiple children, use geometric mean
        node.r_value = math.exp(sum(math.log(gain) for gain in positive_gains) / len(positive_gains))
    
    # Minimum r value to prevent division by zero
    min_r_value = max(G / 100, 0.001)
    node.r_value = max(node.r_value, min_r_value)
    
    # Apply the basic formula: 2^(G/(k*r))
    exponent = G / (k_value * node.r_value)
    node.estimated_size = safe_power_calculation(2, exponent)

def calculate_all_node_r_values_simplified(arranged_nodes, k_value=1.0):
    """Calculate r values and estimated sizes for all nodes using the simplified approach."""
    for node in arranged_nodes:
        calculate_simplified_node_r_and_estimated_size(node, k_value)

def find_optimal_k(arranged_nodes, k_range=(0.1, 10.0), num_trials=100):
    """
    Find the optimal k value that maximizes the good estimate ratio.
    Uses a grid search followed by a finer search around the best value.
    """
    print(f"  Finding optimal k value...")
    
    # First, do a coarse grid search
    k_values = [k_range[0] + i * (k_range[1] - k_range[0]) / (num_trials - 1) for i in range(num_trials)]
    
    best_k = k_values[0]
    best_ratio = 0.0
    
    # Calculate ratio for each k value
    ratios = []
    for k in k_values:
        ratio = calculate_good_estimate_ratio(arranged_nodes, k)
        ratios.append(ratio)
        if ratio > best_ratio:
            best_ratio = ratio
            best_k = k
    
    # Find the range around the best k for finer search
    best_idx = ratios.index(best_ratio)
    if best_ratio > 0:  # Only do finer search if we found something
        # Define a narrower range around the best k
        k_range_fine = max(0.1, best_k - 0.5), min(10.0, best_k + 0.5)
        k_values_fine = [k_range_fine[0] + i * (k_range_fine[1] - k_range_fine[0]) / 49 for i in range(50)]
        
        # Finer search
        for k in k_values_fine:
            ratio = calculate_good_estimate_ratio(arranged_nodes, k)
            if ratio > best_ratio:
                best_ratio = ratio
                best_k = k
    
    print(f"    Optimal k = {best_k:.3f}, Good estimate ratio = {best_ratio:.3f}")
    return best_k, best_ratio

def generate_simplified_tree_output(arranged_nodes, instance_name):
    """Generate formatted tree output using the simplified approach."""
    output_lines = []
    
    # First, find the optimal k value for this instance
    optimal_k, best_ratio = find_optimal_k(arranged_nodes)
    
    # Calculate node-specific r values and estimated sizes using optimal k
    calculate_all_node_r_values_simplified(arranged_nodes, optimal_k)
    
    output_lines.append("="*80)
    output_lines.append(f"SIMPLIFIED Tree Information for {instance_name}")
    output_lines.append("="*80)
    output_lines.append(f"Total nodes: {len(arranged_nodes)}")
    output_lines.append(f"Optimal k value: {optimal_k:.3f}")
    output_lines.append(f"Good estimate ratio: {best_ratio:.3f}")
    output_lines.append("")
    output_lines.append("NOTE: SIMPLIFIED approach using only basic formula: 2^(G/(k*r))")
    output_lines.append("- No children: estimated_size = 1")
    output_lines.append("- One child: r = child_gain")
    output_lines.append("- Two children: r = sqrt(gain1 * gain2)")
    output_lines.append("- Multiple children: r = geometric_mean(all_positive_gains)")
    output_lines.append("- All cases: est_size = 2^(G/(k*r)) (no special cases for gain > G)")
    output_lines.append(f"- Good estimate criterion: 0.5 ≤ est/real ≤ 2.0 (excluding trivial nodes)")
    output_lines.append("")
    
    # Calculate estimation quality statistics
    total_nodes = len(arranged_nodes)
    trivial_nodes = sum(1 for node in arranged_nodes if node.subtree_count == 1)
    non_trivial_nodes = total_nodes - trivial_nodes
    
    good_estimates = 0
    valid_comparisons = 0
    all_ratios = []
    
    for node in arranged_nodes:
        if node.subtree_count > 1 and node.estimated_size > 1.1:
            valid_comparisons += 1
            ratio = node.estimated_size / node.subtree_count
            all_ratios.append(ratio)
            if 0.5 <= ratio <= 2.0:
                good_estimates += 1
    
    output_lines.append("Estimation Quality Summary:")
    output_lines.append(f"  Total nodes: {total_nodes}")
    output_lines.append(f"  Trivial nodes (real=1): {trivial_nodes}")
    output_lines.append(f"  Non-trivial nodes: {non_trivial_nodes}")
    output_lines.append(f"  Valid comparisons: {valid_comparisons}")
    output_lines.append(f"  Good estimates: {good_estimates}")
    output_lines.append(f"  Good estimate ratio: {good_estimates/valid_comparisons:.3f}" if valid_comparisons > 0 else "  Good estimate ratio: N/A")
    
    if all_ratios:
        avg_ratio = sum(all_ratios) / len(all_ratios)
        median_ratio = sorted(all_ratios)[len(all_ratios)//2]
        output_lines.append(f"  Average est/real ratio: {avg_ratio:.3f}")
        output_lines.append(f"  Median est/real ratio: {median_ratio:.3f}")
    output_lines.append("")
    
    # Group nodes by level for summary
    level_summary = {}
    for node in arranged_nodes:
        level = node.level
        if level not in level_summary:
            level_summary[level] = {'count': 0, 'avg_r': [], 'avg_est_size': [], 'avg_real_size': []}
        level_summary[level]['count'] += 1
        if node.r_value is not None:
            level_summary[level]['avg_r'].append(node.r_value)
        level_summary[level]['avg_est_size'].append(node.estimated_size)
        level_summary[level]['avg_real_size'].append(node.subtree_count)
    
    output_lines.append("Summary by Level:")
    for level in sorted(level_summary.keys()):
        data = level_summary[level]
        avg_r = sum(data['avg_r']) / len(data['avg_r']) if data['avg_r'] else "N/A"
        avg_est_size = sum(data['avg_est_size']) / len(data['avg_est_size'])
        avg_real_size = sum(data['avg_real_size']) / len(data['avg_real_size'])
        r_count = len(data['avg_r'])
        
        if isinstance(avg_r, float):
            output_lines.append(f"  Level {level}: {data['count']} nodes, avg r = {avg_r:.6f} ({r_count} nodes with r)")
        else:
            output_lines.append(f"  Level {level}: {data['count']} nodes, avg r = {avg_r} ({r_count} nodes with r)")
        output_lines.append(f"    avg est_size = {avg_est_size:.2f}, avg real_size = {avg_real_size:.2f}")
    output_lines.append("")
    
    current_level = -1
    for node in arranged_nodes:
        # Add level separator
        if node.level != current_level:
            if current_level != -1:
                output_lines.append("")
            output_lines.append(f"LEVEL {node.level}:")
            output_lines.append("-" * 20)
            current_level = node.level
        
        # Node information
        parent_idx = node.parent.idx if node.parent else "None"
        children_idxs = [child.idx for child in node.children]
        
        # Calculate G for this node
        G = node.data['ub'] - node.data['final_lb']
        
        # Calculate actual LB gain and gap for this node
        node_lb_gain = node.data['final_lb'] - node.data['initial_lb']
        node_gap = node.data['ub'] - node.data['final_lb']
        
        # Calculate estimation quality for this node
        est_real_ratio = node.estimated_size / node.subtree_count if node.subtree_count > 0 else float('inf')
        is_good_estimate = (0.5 <= est_real_ratio <= 2.0) if node.subtree_count > 1 and node.estimated_size > 1.1 else "N/A"
        
        output_lines.append(f"Node {node.idx} (Level {node.level}):")
        output_lines.append(f"  Parent: {parent_idx}")
        output_lines.append(f"  Children: {children_idxs}")
        output_lines.append(f"  Real Subtree Count: {node.subtree_count}")
        output_lines.append(f"  Estimated Subtree Size: {node.estimated_size:.2f}")
        output_lines.append(f"  Est/Real Ratio: {est_real_ratio:.3f}")
        output_lines.append(f"  Good Estimate: {is_good_estimate}")
        
        if node.r_value is not None:
            output_lines.append(f"  Node-specific r: {node.r_value:.6f}")
            output_lines.append(f"  k*r: {optimal_k * node.r_value:.6f}")
        else:
            output_lines.append(f"  Node-specific r: N/A (no children or no positive gains)")
        
        output_lines.append(f"  G (Optimality Gap): {G:.6f}")
        output_lines.append(f"  Node LB Gain: {node_lb_gain:.6f}")
        output_lines.append(f"  Node Gap: {node_gap:.6f} {'(feasible)' if node_gap > 0 else '(infeasible)'}")
        
        # Show children's gains if any
        if node.children:
            output_lines.append(f"  Children LB Gains:")
            for child in node.children:
                child_gain = child.data['final_lb'] - child.data['initial_lb']
                # Note: In simplified version, we don't care about >G comparison
                output_lines.append(f"    Child {child.idx}: {child_gain:.6f}")
        
        output_lines.append(f"  Tree Size: {node.data['tree_size']}")
        output_lines.append(f"  Has Edge Evaluation Marker: {node.data['has_marker']}")
        output_lines.append(f"  Initial LB: {node.data['initial_lb']:.6f}")
        output_lines.append(f"  Final LB: {node.data['final_lb']:.6f}")
        output_lines.append(f"  UB: {node.data['ub']:.6f}")
        output_lines.append(f"  Gap: {node.data['gap']:.4f}%")
        output_lines.append(f"  Value: {node.data['value']}")
        
        output_lines.append(f"  Branches ({len(node.data['branches'])}):")
        for j, (edge, direction) in enumerate(node.data['branches']):
            prefix_indicator = "├─" if j < len(node.branch_prefix) else "└─"
            output_lines.append(f"    {prefix_indicator} {j+1}. Edge {edge}: {direction}")
        
        output_lines.append("")
    
    return "\n".join(output_lines)

def extract_instance_name_from_tree_file(filepath):
    """Extract instance name from tree data file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # Look for instance name in the header
            match = re.search(r'Tree Information Details for (.+)', first_line)
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    
    # Fallback: use filename without extension
    return os.path.splitext(os.path.basename(filepath))[0]

def generate_structured_filename(instance_name):
    """Generate structured tree filename."""
    # Extract numbers from instance name
    numbers = re.findall(r'\d+', instance_name)
    if len(numbers) >= 2:
        return f"structured_tree_dg_{numbers[-2]}_{numbers[-1]}.txt"
    else:
        clean_name = re.sub(r'[^\w\d_]', '_', instance_name)
        return f"structured_tree_{clean_name}.txt"

def main():
    # Get input folder
    input_folder = input("Enter the folder path containing tree data files (or press Enter for 'tree_details'): ").strip()
    if not input_folder:
        input_folder = "tree_details"
    
    # Find all tree data files
    tree_files = glob.glob(os.path.join(input_folder, "tree_*.txt"))
    
    if not tree_files:
        print(f"No tree_*.txt files found in {input_folder}")
        return
    
    print(f"Found {len(tree_files)} tree data files")
    
    # Create output directory
    output_dir = "structured_trees_selflog_kmax_simpler"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Process each file
    processed_count = 0
    for filepath in tree_files:
        print(f"Processing: {os.path.basename(filepath)}")
        
        # Extract instance name
        instance_name = extract_instance_name_from_tree_file(filepath)
        
        # Parse tree data
        nodes = parse_tree_data_file(filepath)
        
        if not nodes:
            print(f"  -> No tree data found in {os.path.basename(filepath)}")
            continue
        
        # Build tree structure
        root = build_tree_structure(nodes)
        
        if not root:
            print(f"  -> Could not build tree structure for {os.path.basename(filepath)}")
            continue
        
        # Arrange nodes by level
        arranged_nodes = arrange_nodes_by_level(root)
        
        # Generate output filename
        output_filename = generate_structured_filename(instance_name)
        output_path = os.path.join(output_dir, output_filename)
        
        # Generate simplified structured output
        structured_output = generate_simplified_tree_output(arranged_nodes, instance_name)
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(structured_output)
            
            processed_count += 1
            print(f"  -> Created: {output_filename} ({len(arranged_nodes)} nodes, {root.subtree_count} total subtree)")
            
        except Exception as e:
            print(f"  -> Error writing {output_filename}: {e}")
    
    print(f"\nProcessing complete! {processed_count} files processed successfully.")
    print(f"Simplified tree files saved in: {output_dir}/")

if __name__ == "__main__":
    main()