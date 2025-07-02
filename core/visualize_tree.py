#!/usr/bin/env python3
"""
Tree Visualization Module

Creates interactive visualizations of branch-and-bound trees with estimation quality analysis.
Generates matplotlib plots with color-coded nodes and comprehensive statistics.

Author: Heinrich (Refined)
"""

import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict


# Configuration constants
GOOD_ESTIMATE_RANGE = (0.5, 2.0)
NODE_SIZE_BASE = 200
NODE_SIZE_SCALE = 800


class TreeNodeData:
    """Represents tree node data for visualization."""
    
    def __init__(self, node_id, level, parent_id, children_ids, 
                 subtree_count, estimated_subtree_size):
        self.node_id = node_id
        self.level = level
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.subtree_count = subtree_count
        self.estimated_subtree_size = estimated_subtree_size
        self.gap_percent = None


def load_structured_tree(file_path):
    """
    Parse structured tree output file to extract node information.
    
    Args:
        file_path (str): Path to structured tree file
        
    Returns:
        tuple: (nodes_dict, instance_name)
    """
    tree_nodes = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}, "Unknown"
    
    # Extract instance name
    instance_match = re.search(r'Tree Information for (.+)', content)
    instance_name = instance_match.group(1) if instance_match else "Unknown"
    
    # Parse each node section
    node_pattern = r'Node (\d+) \(Level (\d+)\):(.*?)(?=Node \d+|$)'
    node_matches = re.findall(node_pattern, content, re.DOTALL)
    
    for node_id_str, level_str, node_content in node_matches:
        node_id = int(node_id_str)
        level = int(level_str)
        
        # Extract parent
        parent_match = re.search(r'Parent: (\w+)', node_content)
        parent_id = int(parent_match.group(1)) if parent_match and parent_match.group(1) != 'None' else None
        
        # Extract children
        children_match = re.search(r'Children: \[(.*?)\]', node_content)
        if children_match:
            children_str = children_match.group(1).strip()
            if children_str:
                children_ids = [int(x.strip()) for x in children_str.split(',')]
            else:
                children_ids = []
        else:
            children_ids = []
        
        # Extract subtree count
        subtree_match = re.search(r'Real Subtree Count: (\d+)', node_content)
        subtree_count = int(subtree_match.group(1)) if subtree_match else 1
        
        # Extract estimated subtree size
        est_match = re.search(r'Estimated Subtree Size: ([\d.]+|inf)', node_content)
        if est_match:
            est_str = est_match.group(1)
            if est_str == 'inf':
                estimated_subtree_size = float('inf')
            else:
                estimated_subtree_size = float(est_str)
        else:
            estimated_subtree_size = 1.0
        
        # Extract gap percentage to identify optimal nodes
        gap_match = re.search(r'Gap: ([-\d.]+)%', node_content)
        gap_percent = float(gap_match.group(1)) if gap_match else None
        
        # Extract branches for positioning
        branch_matches = re.findall(r'(?:├─|└─)\s*\d+\.\s*Edge\s+[\d\-]+:\s*([\+\-])', node_content)
        branches = branch_matches
        
        # Create node data
        node_data = TreeNodeData(
            node_id=node_id,
            level=level,
            parent_id=parent_id,
            children_ids=children_ids,
            subtree_count=subtree_count,
            estimated_subtree_size=estimated_subtree_size
        )
        node_data.gap_percent = gap_percent
        node_data.branches = branches  # Store branch directions
        
        tree_nodes[node_id] = node_data
    
    return tree_nodes, instance_name


def calculate_level_statistics(tree_nodes):
    """Calculate estimation quality statistics for each level (excluding trivial real=1 nodes)."""
    level_stats = defaultdict(lambda: {'good': 0, 'over': 0, 'under': 0, 'inf': 0, 'total': 0})
    
    for node in tree_nodes.values():
        level = node.level
        actual_count = node.subtree_count
        estimated_size = node.estimated_subtree_size
        
        # Only include non-trivial nodes (real > 1)
        if actual_count <= 1:
            continue
            
        level_stats[level]['total'] += 1
        
        if estimated_size == float('inf'):
            level_stats[level]['inf'] += 1
        else:
            ratio = estimated_size / actual_count
            
            if GOOD_ESTIMATE_RANGE[0] <= ratio <= GOOD_ESTIMATE_RANGE[1]:
                level_stats[level]['good'] += 1
            elif ratio > GOOD_ESTIMATE_RANGE[1]:
                level_stats[level]['over'] += 1
            else:
                level_stats[level]['under'] += 1
    
    return dict(level_stats)


def build_networkx_graph(tree_nodes):
    """Create NetworkX graph from node data."""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in tree_nodes.items():
        G.add_node(node_id, 
                  level=node_data.level,
                  subtree_count=node_data.subtree_count,
                  estimated_subtree_size=node_data.estimated_subtree_size)
    
    # Add edges
    for node_id, node_data in tree_nodes.items():
        if node_data.parent_id is not None:
            G.add_edge(node_data.parent_id, node_id)
    
    return G


def compute_layout_positions(G, tree_nodes):
    """Calculate positions for nodes respecting branch directions (- left, + right) with overlap prevention."""
    # Try graphviz first if available
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        # Flip y-coordinates and scale for better spacing
        pos = {node: (x * 1.5, -y * 1.2) for node, (x, y) in pos.items()}
        return pos
    except:
        pass
    
    # Custom layout that respects branch directions
    levels = defaultdict(list)
    for node_id, node_data in tree_nodes.items():
        levels[node_data.level].append(node_id)
    
    pos = {}
    max_level = max(levels.keys()) if levels else 0
    min_spacing = 2.5  # Minimum distance between nodes
    
    # Position root at center
    root_nodes = levels.get(0, [])
    if root_nodes:
        pos[root_nodes[0]] = (0, max_level)
    
    # Position each level based on branch directions
    for level in range(1, max_level + 1):
        level_nodes = levels[level]
        if not level_nodes:
            continue
            
        y = max_level - level
        
        # Group nodes by their parent
        parent_groups = defaultdict(list)
        for node_id in level_nodes:
            parent_id = tree_nodes[node_id].parent_id
            parent_groups[parent_id].append(node_id)
        
        # Process each parent group separately (keeps siblings together)
        all_parent_positions = []
        
        for parent_id, children in parent_groups.items():
            if parent_id is None:
                # Orphan nodes - position at center
                parent_positions = []
                for i, child_id in enumerate(children):
                    parent_positions.append((i * min_spacing, child_id))
                all_parent_positions.append(parent_positions)
                continue
            
            parent_x = pos.get(parent_id, (0, 0))[0]
            
            # Separate children by branch direction within this parent group
            left_children = []  # - branches
            right_children = [] # + branches
            
            for child_id in children:
                child_node = tree_nodes[child_id]
                # Get the last branch direction (the one that led to this node)
                if hasattr(child_node, 'branches') and child_node.branches:
                    # Get the last (most recent) branch direction
                    last_direction = child_node.branches[-1]
                    if last_direction == '-':
                        left_children.append(child_id)
                    elif last_direction == '+':
                        right_children.append(child_id)
                    else:
                        # Fallback: treat as right branch
                        right_children.append(child_id)
                else:
                    # No branch info available, treat as right branch
                    right_children.append(child_id)
            
            # Create positions for this parent's children
            parent_positions = []
            
            # Position left children (- branches) to the left of parent
            if left_children:
                for i, child_id in enumerate(sorted(left_children, reverse=True)):
                    x_offset = -(i + 1) * 1.2  # Closer spacing within family
                    parent_positions.append((parent_x + x_offset, child_id))
            
            # Position right children (+ branches) to the right of parent
            if right_children:
                for i, child_id in enumerate(sorted(right_children)):
                    x_offset = (i + 1) * 1.2  # Closer spacing within family
                    parent_positions.append((parent_x + x_offset, child_id))
            
            # Sort this parent's children by x position (left to right)
            parent_positions.sort(key=lambda x: x[0])
            all_parent_positions.append(parent_positions)
        
        # Now arrange parent groups to avoid overlaps between families
        # Sort parent groups by their leftmost child's position
        all_parent_positions.sort(key=lambda group: min(x for x, _ in group))
        
        # Adjust positions to prevent overlaps between parent groups
        final_positions = []
        current_x = float('-inf')
        
        for parent_group in all_parent_positions:
            # Check if this group overlaps with previous groups
            group_min_x = min(x for x, _ in parent_group)
            
            if group_min_x - current_x < min_spacing:
                # Need to shift this entire group right
                shift = (current_x + min_spacing) - group_min_x
                adjusted_group = [(x + shift, node_id) for x, node_id in parent_group]
            else:
                adjusted_group = parent_group
            
            # Update the rightmost position
            current_x = max(x for x, _ in adjusted_group)
            final_positions.extend(adjusted_group)
        
        # Center the entire level
        if final_positions:
            min_x = min(x for x, _ in final_positions)
            max_x = max(x for x, _ in final_positions)
            center_offset = -(min_x + max_x) / 2
            
            # Apply centering and set final positions
            for x, node_id in final_positions:
                pos[node_id] = (x + center_offset, y)
    
    return pos


def calculate_estimation_quality(tree_nodes):
    """Calculate estimation quality statistics, both including and excluding nodes with real=1."""
    # All nodes statistics
    all_good = 0
    all_over = 0
    all_under = 0
    all_inf = 0
    
    # Excluding real=1 nodes statistics  
    excl_good = 0
    excl_over = 0
    excl_under = 0
    excl_inf = 0
    
    for node in tree_nodes.values():
        actual_count = node.subtree_count
        estimated_size = node.estimated_subtree_size
        
        if estimated_size == float('inf'):
            all_inf += 1
            if actual_count > 1:
                excl_inf += 1
        else:
            ratio = estimated_size / max(actual_count, 1)
            
            # Categorize estimation quality
            if GOOD_ESTIMATE_RANGE[0] <= ratio <= GOOD_ESTIMATE_RANGE[1]:
                all_good += 1
                if actual_count > 1:
                    excl_good += 1
            elif ratio > GOOD_ESTIMATE_RANGE[1]:
                all_over += 1
                if actual_count > 1:
                    excl_over += 1
            else:
                all_under += 1
                if actual_count > 1:
                    excl_under += 1
    
    # Calculate totals
    all_total = len(tree_nodes)
    all_finite = all_total - all_inf
    
    excl_total = sum(1 for node in tree_nodes.values() if node.subtree_count > 1)
    excl_finite = excl_total - excl_inf
    
    return {
        'all': {
            'total': all_total,
            'finite': all_finite,
            'good': all_good,
            'over': all_over,
            'under': all_under,
            'inf': all_inf
        },
        'excluding_real_1': {
            'total': excl_total,
            'finite': excl_finite,
            'good': excl_good,
            'over': excl_over,
            'under': excl_under,
            'inf': excl_inf
        }
    }


def determine_node_appearance(tree_nodes):
    """Calculate node colors and sizes based on subtree counts and estimates."""
    # Get ranges for normalization
    actual_counts = [node.subtree_count for node in tree_nodes.values()]
    max_actual = max(actual_counts) if actual_counts else 1
    
    colors = []
    sizes = []
    
    # Process nodes in sorted order by node_id to match node_list
    for node_id in sorted(tree_nodes.keys()):
        node_data = tree_nodes[node_id]
        
        # Color based on ratio of estimated to real
        if node_data.estimated_subtree_size == float('inf'):
            colors.append('red')
        else:
            ratio = node_data.estimated_subtree_size / max(node_data.subtree_count, 1)
            if ratio > GOOD_ESTIMATE_RANGE[1]:
                colors.append('orange')  # Over-estimated
            elif ratio < GOOD_ESTIMATE_RANGE[0]:
                colors.append('lightblue')  # Under-estimated
            else:
                colors.append('lightgreen')  # Good estimate
        
        # Size based on real subtree count
        normalized_size = (node_data.subtree_count / max_actual) * NODE_SIZE_SCALE + NODE_SIZE_BASE
        sizes.append(normalized_size)
    
    return colors, sizes


def generate_node_labels(tree_nodes):
    """Create compact labels with real and est on separate lines, hide trivial (1, 1.0) nodes."""
    labels = {}
    
    for node_id, node_data in tree_nodes.items():
        actual_count = node_data.subtree_count
        
        if node_data.estimated_subtree_size == float('inf'):
            est_str = "∞"
            # Always show infinite estimates
            labels[node_id] = f"{actual_count}\n{est_str}"
        else:
            estimated_size = node_data.estimated_subtree_size
            
            # Hide labels for trivial nodes: real=1 and est≈1.0
            if actual_count == 1 and abs(estimated_size - 1.0) < 0.1:
                labels[node_id] = ""  # Empty label (node will still be visible)
            else:
                est_str = f"{estimated_size:.1f}"
                labels[node_id] = f"{actual_count}\n{est_str}"
    
    return labels


def add_level_stats(tree_nodes, pos, ax):
    """Add level-wise statistics text to the plot."""
    level_stats = calculate_level_statistics(tree_nodes)
    
    if not level_stats:
        return
    
    # Find the position range for each level
    levels = defaultdict(list)
    for node_id, node_data in tree_nodes.items():
        if node_id in pos:
            levels[node_data.level].append(pos[node_id])
    
    # Add text for each level with statistics
    for level, stats in level_stats.items():
        if stats['total'] == 0:
            continue
            
        # Calculate good estimate percentage
        finite_total = stats['total'] - stats['inf']
        if finite_total > 0:
            good_percent = (stats['good'] / finite_total) * 100
        else:
            good_percent = 0
        
        # Find the rightmost position at this level
        level_positions = levels.get(level, [])
        if level_positions:
            max_x = max(x for x, y in level_positions)
            level_y = level_positions[0][1]  # All nodes at same level have same y
            
            # Create statistics text
            if stats['inf'] > 0:
                stats_text = f"L{level}: {good_percent:.1f}% good\n({stats['good']}/{finite_total}, {stats['inf']} inf)"
            else:
                stats_text = f"L{level}: {good_percent:.1f}% good\n({stats['good']}/{finite_total})"
            
            # Position text to the right of the level
            text_x = max_x + 2.0
            ax.text(text_x, level_y, stats_text, 
                   fontsize=9, fontweight='bold',
                   verticalalignment='center', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                   family='monospace')


def create_tree_plot(file_path, output_dir=None, show_plot=True):
    """
    Main function to visualize tree structure.
    
    Args:
        file_path (str): Path to structured tree file
        output_dir (str): Directory to save plot (optional)
        show_plot (bool): Whether to display the plot interactively
        
    Returns:
        dict: Summary statistics about the tree
    """
    print(f"Processing file: {file_path}")
    
    # Parse the structured tree file
    tree_nodes, instance_name = load_structured_tree(file_path)
    
    if not tree_nodes:
        print(f"No nodes found in {file_path}")
        return {}
    
    print(f"Visualizing tree for {instance_name} with {len(tree_nodes)} nodes")
    
    # Create graph
    G = build_networkx_graph(tree_nodes)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate layout
    pos = compute_layout_positions(G, tree_nodes)
    print(f"Calculated positions for {len(pos)} nodes")
    
    # Get colors and sizes
    colors, sizes = determine_node_appearance(tree_nodes)
    print(f"Generated {len(colors)} colors and {len(sizes)} sizes")
    
    # Create labels
    labels = generate_node_labels(tree_nodes)
    
    # Create the plot
    print("Creating matplotlib figure...")
    plt.figure(figsize=(18, 12))  # Increased width for level statistics
    
    # Draw the tree with better edge styling
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', alpha=0.7, width=1.5,
                          connectionstyle="arc3,rad=0.1")  # Slight curve for better visibility
    
    # Draw nodes first (background layer)
    node_list = sorted(tree_nodes.keys())  # Ensure consistent ordering
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.9, linewidths=2, edgecolors='black')
    
    # Draw stars for optimal nodes (0% gap)
    optimal_nodes = [node_id for node_id, node_data in tree_nodes.items() 
                    if hasattr(node_data, 'gap_percent') and node_data.gap_percent is not None and abs(node_data.gap_percent) < 0.001]
    
    if optimal_nodes:
        optimal_positions = [pos[node_id] for node_id in optimal_nodes]
        optimal_x = [pos[0] for pos in optimal_positions]
        optimal_y = [pos[1] for pos in optimal_positions]
        
        # Draw gold stars on optimal nodes
        plt.scatter(optimal_x, optimal_y, s=150, c='gold', marker='*', 
                   edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
        
        print(f"Found {len(optimal_nodes)} optimal nodes (0% gap): {optimal_nodes}")
    
    # Draw labels with better positioning (left side of nodes)
    label_pos = {node: (x - 0.3, y) for node, (x, y) in pos.items()}  # Offset to left
    nx.draw_networkx_labels(G, label_pos, labels, font_size=9, font_weight='bold',
                          font_color='darkblue', verticalalignment='center',
                          horizontalalignment='right',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add level-wise statistics to the plot
    add_level_stats(tree_nodes, pos, plt.gca())
    
    # Calculate and display estimation quality in top-right corner
    quality_stats = calculate_estimation_quality(tree_nodes)
    
    # Create text for estimation quality
    all_stats = quality_stats['all']
    excl_stats = quality_stats['excluding_real_1']
    
    if all_stats['finite'] > 0:
        quality_text = "Overall Estimation Quality:\n"
        quality_text += f"Good estimates: {all_stats['good']} ({all_stats['good']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Over-estimates: {all_stats['over']} ({all_stats['over']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Under-estimates: {all_stats['under']} ({all_stats['under']/all_stats['finite']*100:.1f}%)\n"
        
        if excl_stats['finite'] > 0:
            quality_text += f"\nExcluding real=1 nodes:\n"
            quality_text += f"Good estimates: {excl_stats['good']} ({excl_stats['good']/excl_stats['finite']*100:.1f}%)\n"
            quality_text += f"Over-estimates: {excl_stats['over']} ({excl_stats['over']/excl_stats['finite']*100:.1f}%)\n"
            quality_text += f"Under-estimates: {excl_stats['under']} ({excl_stats['under']/excl_stats['finite']*100:.1f}%)"
        
        # Add text box in top-right corner
        plt.text(0.98, 0.98, quality_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                family='monospace')
    
    # Create legend including optimal node marker
    legend_elements = create_legend_elements()
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Set title and formatting
    plt.title(f"Tree Structure: {instance_name}\n"
             f"Node Format: (Real, Est), Size ∝ Real Count, Color = Quality\n"
             f"Level Stats: % Good Estimates (excluding trivial real=1 nodes)", 
             fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"tree_visualization_{instance_name.replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, filename)
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    # Show plot
    if show_plot:
        print("Attempting to show plot...")
        plt.show()
    else:
        print("Skipping plot display (show_plot=False)")
        plt.close()
    
    # Print summary statistics
    print_tree_statistics(tree_nodes, instance_name)
    
    # Return summary statistics
    return {
        'instance_name': instance_name,
        'total_nodes': len(tree_nodes),
        'optimal_nodes': len(optimal_nodes),
        'quality_stats': quality_stats,
        'level_stats': calculate_level_statistics(tree_nodes)
    }


def create_legend_elements():
    """Create legend elements for the tree visualization."""
    legend_elements = [
        mpatches.Patch(color='lightgreen', label=f'Good Estimate ({GOOD_ESTIMATE_RANGE[0]}x ≤ Est/Real ≤ {GOOD_ESTIMATE_RANGE[1]}x)'),
        mpatches.Patch(color='orange', label=f'Over-estimated (Est/Real > {GOOD_ESTIMATE_RANGE[1]}x)'),
        mpatches.Patch(color='lightblue', label=f'Under-estimated (Est/Real < {GOOD_ESTIMATE_RANGE[0]}x)'),
        mpatches.Patch(color='red', label='Infinite Estimate'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=12, label='Optimal Node (0% gap)')
    ]
    return legend_elements


def print_tree_statistics(tree_nodes, instance_name):
    """Print summary statistics about the tree."""
    print(f"\n--- Tree Statistics for {instance_name} ---")
    
    actual_counts = [node.subtree_count for node in tree_nodes.values()]
    est_counts = [node.estimated_subtree_size for node in tree_nodes.values() 
                 if node.estimated_subtree_size != float('inf')]
    
    # Count optimal nodes
    optimal_count = sum(1 for node in tree_nodes.values() 
                       if hasattr(node, 'gap_percent') and node.gap_percent is not None and abs(node.gap_percent) < 0.001)
    
    print(f"Total nodes: {len(tree_nodes)}")
    print(f"Optimal nodes (0% gap): {optimal_count}")
    print(f"Real subtree counts - Min: {min(actual_counts)}, Max: {max(actual_counts)}, Avg: {np.mean(actual_counts):.1f}")
    
    if est_counts:
        print(f"Estimated subtree sizes - Min: {min(est_counts):.1f}, Max: {max(est_counts):.1f}, Avg: {np.mean(est_counts):.1f}")
    
    # Print detailed quality statistics
    quality_stats = calculate_estimation_quality(tree_nodes)
    all_stats = quality_stats['all']
    excl_stats = quality_stats['excluding_real_1']
    
    print(f"Nodes with infinite estimates: {all_stats['inf']}")
    
    if all_stats['finite'] > 0:
        print(f"Estimation quality (all nodes):")
        print(f"  Good estimates: {all_stats['good']} ({all_stats['good']/all_stats['finite']*100:.1f}%)")
        print(f"  Over-estimates: {all_stats['over']} ({all_stats['over']/all_stats['finite']*100:.1f}%)")
        print(f"  Under-estimates: {all_stats['under']} ({all_stats['under']/all_stats['finite']*100:.1f}%)")
    
    if excl_stats['finite'] > 0:
        print(f"Estimation quality (excluding real=1 nodes):")
        print(f"  Good estimates: {excl_stats['good']} ({excl_stats['good']/excl_stats['finite']*100:.1f}%)")
        print(f"  Over-estimates: {excl_stats['over']} ({excl_stats['over']/excl_stats['finite']*100:.1f}%)")
        print(f"  Under-estimates: {excl_stats['under']} ({excl_stats['under']/excl_stats['finite']*100:.1f}%)")
    
    # Print level-wise statistics
    level_stats = calculate_level_statistics(tree_nodes)
    if level_stats:
        print(f"\nLevel-wise estimation quality (excluding real=1 nodes):")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            if stats['total'] > 0:
                finite_total = stats['total'] - stats['inf']
                if finite_total > 0:
                    good_percent = (stats['good'] / finite_total) * 100
                    print(f"  Level {level}: {good_percent:.1f}% good ({stats['good']}/{finite_total})", end="")
                    if stats['inf'] > 0:
                        print(f", {stats['inf']} infinite")
                    else:
                        print()


def process_multiple_tree_files(input_path, output_dir=None, show_plots=False):
    """
    Process multiple structured tree files for visualization.
    
    Args:
        input_path (str or Path): Path to file or directory containing structured tree files
        output_dir (str): Directory to save plots (optional)
        show_plots (bool): Whether to display plots interactively
        
    Returns:
        list: List of summary statistics for each processed file
    """
    input_path = Path(input_path)
    results = []
    
    if input_path.is_file():
        # Single file
        stats = create_tree_plot(str(input_path), output_dir, show_plots)
        results.append(stats)
    elif input_path.is_dir():
        # Directory - process all structured tree files
        tree_files = list(input_path.glob("structured_tree_*.txt"))
        
        if not tree_files:
            print(f"No structured_tree_*.txt files found in {input_path}")
            return results
        
        print(f"Found {len(tree_files)} structured tree files")
        
        for file_path in sorted(tree_files):
            print(f"\nProcessing: {file_path.name}")
            stats = create_tree_plot(str(file_path), output_dir, False)  # Don't show individual plots
            results.append(stats)
        
        print(f"\nProcessed {len(tree_files)} files")
        if output_dir:
            print(f"All visualizations saved to: {output_dir}")
    else:
        print(f"Invalid path: {input_path}")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Visualize tree structure from structured tree files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 visualize_tree.py structured_tree_file.txt
  python3 visualize_tree.py structured_trees/ --output-dir plots/ 
  python3 visualize_tree.py structured_tree_file.txt --no-show
        """
    )
    
    parser.add_argument('input_path', help='Path to structured tree file or directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for saving plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    # Process the files
    results = process_multiple_tree_files(
        input_path=args.input_path,
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )
    
    # Print overall summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Processed {len(results)} tree files")
        
        total_nodes = sum(r.get('total_nodes', 0) for r in results)
        total_optimal = sum(r.get('optimal_nodes', 0) for r in results)
        
        print(f"Total nodes across all trees: {total_nodes}")
        print(f"Total optimal nodes: {total_optimal}")
        
        if results:
            avg_quality = []
            for r in results:
                quality = r.get('quality_stats', {})
                excl_stats = quality.get('excluding_real_1', {})
                if excl_stats.get('finite', 0) > 0:
                    ratio = excl_stats['good'] / excl_stats['finite']
                    avg_quality.append(ratio)
            
            if avg_quality:
                print(f"Average good estimate ratio: {np.mean(avg_quality):.3f}")


if __name__ == "__main__":
    main()