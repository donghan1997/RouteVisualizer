#!/usr/bin/env python3
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import argparse
from pathlib import Path
from collections import defaultdict

class TreeNodeData:
    def __init__(self, node_id, level, parent_id, children_ids, 
                 subtree_count, estimated_subtree_size):
        self.node_id = node_id
        self.level = level
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.subtree_count = subtree_count
        self.estimated_subtree_size = estimated_subtree_size
        self.gap_percent = None

def parse_structured_tree_file(file_path):
    """Parse structured tree output file to extract node information."""
    nodes = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}, "Unknown"
    
    # Extract instance name
    instance_match = re.search(r'Structured Tree Information for (.+)', content)
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
        subtree_match = re.search(r'Subtree Count: (\d+)', node_content)
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
        
        nodes[node_id] = node_data
    
    return nodes, instance_name

def calculate_level_statistics(nodes):
    """Calculate estimation quality statistics for each level (excluding trivial real=1 nodes)."""
    level_stats = defaultdict(lambda: {'good': 0, 'over': 0, 'under': 0, 'inf': 0, 'total': 0})
    
    for node in nodes.values():
        level = node.level
        real_count = node.subtree_count
        est_size = node.estimated_subtree_size
        
        # Only include non-trivial nodes (real > 1)
        if real_count <= 1:
            continue
            
        level_stats[level]['total'] += 1
        
        if est_size == float('inf'):
            level_stats[level]['inf'] += 1
        else:
            ratio = est_size / real_count
            
            if 0.5 <= ratio <= 2:
                level_stats[level]['good'] += 1
            elif ratio > 2:
                level_stats[level]['over'] += 1
            else:
                level_stats[level]['under'] += 1
    
    return dict(level_stats)

def create_tree_graph(nodes):
    """Create NetworkX graph from node data."""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in nodes.items():
        G.add_node(node_id, 
                  level=node_data.level,
                  subtree_count=node_data.subtree_count,
                  estimated_subtree_size=node_data.estimated_subtree_size)
    
    # Add edges
    for node_id, node_data in nodes.items():
        if node_data.parent_id is not None:
            G.add_edge(node_data.parent_id, node_id)
    
    return G

def calculate_node_positions(G, nodes):
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
    for node_id, node_data in nodes.items():
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
            parent_id = nodes[node_id].parent_id
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
                child_node = nodes[child_id]
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

def calculate_estimation_quality(nodes):
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
    
    for node in nodes.values():
        real_count = node.subtree_count
        est_size = node.estimated_subtree_size
        
        if est_size == float('inf'):
            all_inf += 1
            if real_count > 1:
                excl_inf += 1
        else:
            ratio = est_size / max(real_count, 1)
            
            # Categorize estimation quality
            if 0.5 <= ratio <= 2:
                all_good += 1
                if real_count > 1:
                    excl_good += 1
            elif ratio > 2:
                all_over += 1
                if real_count > 1:
                    excl_over += 1
            else:
                all_under += 1
                if real_count > 1:
                    excl_under += 1
    
    # Calculate totals
    all_total = len(nodes)
    all_finite = all_total - all_inf
    
    excl_total = sum(1 for node in nodes.values() if node.subtree_count > 1)
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

def get_node_colors_and_sizes(nodes):
    """Calculate node colors and sizes based on subtree counts and estimates."""
    # Get ranges for normalization
    real_counts = [node.subtree_count for node in nodes.values()]
    max_real = max(real_counts) if real_counts else 1
    
    colors = []
    sizes = []
    
    # Process nodes in sorted order by node_id to match node_list
    for node_id in sorted(nodes.keys()):
        node_data = nodes[node_id]
        
        # Color based on ratio of estimated to real
        if node_data.estimated_subtree_size == float('inf'):
            colors.append('red')
        else:
            ratio = node_data.estimated_subtree_size / max(node_data.subtree_count, 1)
            if ratio > 2:
                colors.append('orange')  # Over-estimated
            elif ratio < 0.5:
                colors.append('lightblue')  # Under-estimated
            else:
                colors.append('lightgreen')  # Good estimate
        
        # Size based on real subtree count
        normalized_size = (node_data.subtree_count / max_real) * 800 + 200
        sizes.append(normalized_size)
    
    return colors, sizes

def create_node_labels(nodes):
    """Create compact labels with real and est on separate lines, hide trivial (1, 1.0) nodes."""
    labels = {}
    
    for node_id, node_data in nodes.items():
        real_count = node_data.subtree_count
        
        if node_data.estimated_subtree_size == float('inf'):
            est_str = "∞"
            # Always show infinite estimates
            labels[node_id] = f"{real_count}\n{est_str}"
        else:
            est_size = node_data.estimated_subtree_size
            
            # Hide labels for trivial nodes: real=1 and est≈1.0
            if real_count == 1 and abs(est_size - 1.0) < 0.1:
                labels[node_id] = ""  # Empty label (node will still be visible)
            else:
                est_str = f"{est_size:.1f}"
                labels[node_id] = f"{real_count}\n{est_str}"
    
    return labels

def add_level_statistics_to_plot(nodes, pos):
    """Add level-wise statistics text to the plot."""
    level_stats = calculate_level_statistics(nodes)
    
    if not level_stats:
        return
    
    # Find the position range for each level
    levels = defaultdict(list)
    for node_id, node_data in nodes.items():
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
            plt.text(text_x, level_y, stats_text, 
                    fontsize=9, fontweight='bold',
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                    family='monospace')

def visualize_tree(file_path, output_dir=None, show_plot=True):
    """Main function to visualize tree structure."""
    
    print(f"Processing file: {file_path}")
    
    # Parse the structured tree file
    nodes, instance_name = parse_structured_tree_file(file_path)
    
    if not nodes:
        print(f"No nodes found in {file_path}")
        return
    
    print(f"Visualizing tree for {instance_name} with {len(nodes)} nodes")
    
    # Create graph
    G = create_tree_graph(nodes)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate layout
    pos = calculate_node_positions(G, nodes)
    print(f"Calculated positions for {len(pos)} nodes")
    
    # Get colors and sizes
    colors, sizes = get_node_colors_and_sizes(nodes)
    print(f"Generated {len(colors)} colors and {len(sizes)} sizes")
    
    # Create labels
    labels = create_node_labels(nodes)
    
    # Create the plot
    print("Creating matplotlib figure...")
    plt.figure(figsize=(18, 12))  # Increased width for level statistics
    
    # Draw the tree with better edge styling
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', alpha=0.7, width=1.5,
                          connectionstyle="arc3,rad=0.1")  # Slight curve for better visibility
    
    # Draw nodes first (background layer)
    node_list = sorted(nodes.keys())  # Ensure consistent ordering
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.9, linewidths=2, edgecolors='black')
    
    # Draw stars for optimal nodes (0% gap)
    optimal_nodes = [node_id for node_id, node_data in nodes.items() 
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
    add_level_statistics_to_plot(nodes, pos)
    
    # Calculate and display estimation quality in top-right corner
    quality_stats = calculate_estimation_quality(nodes)
    
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
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Good Estimate (0.5x ≤ Est/Real ≤ 2x)'),
        mpatches.Patch(color='orange', label='Over-estimated (Est/Real > 2x)'),
        mpatches.Patch(color='lightblue', label='Under-estimated (Est/Real < 0.5x)'),
        mpatches.Patch(color='red', label='Infinite Estimate'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=12, label='Optimal Node (0% gap)')
    ]
    
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
    
    # Print summary statistics
    print_tree_statistics(nodes, instance_name)

def print_tree_statistics(nodes, instance_name):
    """Print summary statistics about the tree."""
    print(f"\n--- Tree Statistics for {instance_name} ---")
    
    real_counts = [node.subtree_count for node in nodes.values()]
    est_counts = [node.estimated_subtree_size for node in nodes.values() 
                 if node.estimated_subtree_size != float('inf')]
    
    # Count optimal nodes
    optimal_count = sum(1 for node in nodes.values() 
                       if hasattr(node, 'gap_percent') and node.gap_percent is not None and abs(node.gap_percent) < 0.001)
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Optimal nodes (0% gap): {optimal_count}")
    print(f"Real subtree counts - Min: {min(real_counts)}, Max: {max(real_counts)}, Avg: {np.mean(real_counts):.1f}")
    
    if est_counts:
        print(f"Estimated subtree sizes - Min: {min(est_counts):.1f}, Max: {max(est_counts):.1f}, Avg: {np.mean(est_counts):.1f}")
    
    # Print detailed quality statistics
    quality_stats = calculate_estimation_quality(nodes)
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
    level_stats = calculate_level_statistics(nodes)
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

def main():
    parser = argparse.ArgumentParser(description='Visualize tree structure from structured tree files')
    parser.add_argument('input_path', help='Path to structured tree file or directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for saving plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file
        visualize_tree(str(input_path), args.output_dir, not args.no_show)
    elif input_path.is_dir():
        # Directory - process all structured tree files
        tree_files = list(input_path.glob("structured_tree_*.txt"))
        
        if not tree_files:
            print(f"No structured_tree_*.txt files found in {input_path}")
            return
        
        print(f"Found {len(tree_files)} structured tree files")
        
        for file_path in sorted(tree_files):
            print(f"\nProcessing: {file_path.name}")
            visualize_tree(str(file_path), args.output_dir, False)  # Don't show individual plots
        
        print(f"\nProcessed {len(tree_files)} files")
        if args.output_dir:
            print(f"All visualizations saved to: {args.output_dir}")
    else:
        print(f"Invalid path: {input_path}")

if __name__ == "__main__":
    main()