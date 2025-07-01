#!/usr/bin/env python3
"""
Simple Tree Visualization Module

Creates clean visualizations of branch-and-bound trees focusing on tree structure
and node termination reasons (bound pruning, infeasible, integer solution).

Author: Heinrich (Simplified)
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
DEFAULT_NODE_SIZE = 100
NODE_SIZE_SCALE = 200
DEFAULT_FIGURE_SIZE = (14, 10)


class SimpleTreeNodeData:
    """Represents simplified tree node data for visualization."""
    
    def __init__(self, node_id, level, parent_id, children_ids):
        self.node_id = node_id
        self.level = level
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.subtree_count = 1
        self.termination_reason = None  # 'bound_pruned', 'infeasible', 'integer_solution', 'active'
        self.gap_percent = None
        self.is_optimal = False


def load_structured_tree_simple(file_path):
    """
    Parse structured tree output file to extract basic node information.
    
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
        
        # Extract gap percentage to identify optimal nodes
        gap_match = re.search(r'Gap: ([-\d.]+)%', node_content)
        gap_percent = float(gap_match.group(1)) if gap_match else None
        
        # Determine termination reason based on content
        termination_reason = determine_termination_reason(node_content, children_ids)
        
        # Create simplified node data
        node_data = SimpleTreeNodeData(
            node_id=node_id,
            level=level,
            parent_id=parent_id,
            children_ids=children_ids
        )
        node_data.subtree_count = subtree_count
        node_data.gap_percent = gap_percent
        node_data.termination_reason = termination_reason
        node_data.is_optimal = gap_percent is not None and abs(gap_percent) < 0.001
        
        tree_nodes[node_id] = node_data
    
    return tree_nodes, instance_name


def determine_termination_reason(node_content, children_ids):
    """
    Determine why a node was terminated based on the content.
    
    Args:
        node_content (str): Content of the node section
        children_ids (list): List of children node IDs
        
    Returns:
        str: Termination reason
    """
    # If node has children, it's still active (branched)
    if children_ids:
        return 'active'
    
    # Check for specific termination indicators in the content
    content_lower = node_content.lower()
    
    # Check for infeasible
    if 'infeasible' in content_lower:
        return 'infeasible'
    
    # Check for integer solution
    if 'integer' in content_lower or 'optimal' in content_lower:
        return 'integer_solution'
    
    # Check for bound pruning (gap close to 0, or bounded)
    if 'bound' in content_lower or 'prune' in content_lower:
        return 'bound_pruned'
    
    # Extract gap to help determine termination
    gap_match = re.search(r'Gap: ([-\d.]+)%', node_content)
    if gap_match:
        gap = abs(float(gap_match.group(1)))
        if gap < 0.01:  # Very small gap
            return 'bound_pruned'
    
    # Default: assume bound pruned if it's a leaf node
    return 'bound_pruned'


def build_networkx_graph_simple(tree_nodes):
    """Create NetworkX graph from simplified node data."""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in tree_nodes.items():
        G.add_node(node_id, 
                  level=node_data.level,
                  subtree_count=node_data.subtree_count,
                  termination_reason=node_data.termination_reason)
    
    # Add edges
    for node_id, node_data in tree_nodes.items():
        if node_data.parent_id is not None:
            G.add_edge(node_data.parent_id, node_id)
    
    return G


def compute_layout_positions_simple(G, tree_nodes):
    """Calculate positions for nodes using a simple hierarchical layout."""
    # Try graphviz first if available
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        # Flip y-coordinates and scale for better spacing
        pos = {node: (x * 1.2, -y) for node, (x, y) in pos.items()}
        return pos
    except:
        pass
    
    # Fallback: simple level-based layout
    levels = defaultdict(list)
    for node_id, node_data in tree_nodes.items():
        levels[node_data.level].append(node_id)
    
    pos = {}
    max_level = max(levels.keys()) if levels else 0
    
    for level in sorted(levels.keys()):
        level_nodes = levels[level]
        y = max_level - level
        
        # Center nodes horizontally
        total_width = len(level_nodes) * 2
        start_x = -total_width / 2
        
        for i, node_id in enumerate(sorted(level_nodes)):
            x = start_x + i * 2
            pos[node_id] = (x, y)
    
    return pos


def determine_node_appearance_simple(tree_nodes):
    """Calculate node colors and sizes based on termination reasons and subtree counts."""
    # Define colors for different termination reasons
    color_map = {
        'active': 'lightblue',           # Nodes that were branched (have children)
        'bound_pruned': 'lightgreen',    # Nodes pruned by bounds
        'infeasible': 'lightcoral',      # Infeasible nodes
        'integer_solution': 'gold',      # Integer solutions
    }
    
    # Get ranges for size normalization
    actual_counts = [node.subtree_count for node in tree_nodes.values()]
    max_actual = max(actual_counts) if actual_counts else 1
    
    colors = []
    sizes = []
    
    # Process nodes in sorted order by node_id
    for node_id in sorted(tree_nodes.keys()):
        node_data = tree_nodes[node_id]
        
        # Color based on termination reason
        colors.append(color_map.get(node_data.termination_reason, 'lightgray'))
        
        # Size based on subtree count (smaller range)
        normalized_size = (node_data.subtree_count / max_actual) * NODE_SIZE_SCALE + DEFAULT_NODE_SIZE
        sizes.append(normalized_size)
    
    return colors, sizes


def generate_simple_node_labels(tree_nodes):
    """Create simple labels showing only node ID."""
    labels = {}
    
    for node_id, node_data in tree_nodes.items():
        # Only show node ID for clean appearance
        labels[node_id] = str(node_id)
    
    return labels


def create_simple_tree_plot(file_path, output_dir=None, show_plot=True):
    """
    Main function to visualize tree structure in simplified form.
    
    Args:
        file_path (str): Path to structured tree file
        output_dir (str): Directory to save plot (optional)
        show_plot (bool): Whether to display the plot interactively
        
    Returns:
        dict: Basic summary statistics about the tree
    """
    print(f"Processing file: {file_path}")
    
    # Parse the structured tree file
    tree_nodes, instance_name = load_structured_tree_simple(file_path)
    
    if not tree_nodes:
        print(f"No nodes found in {file_path}")
        return {}
    
    print(f"Visualizing simplified tree for {instance_name} with {len(tree_nodes)} nodes")
    
    # Create graph
    G = build_networkx_graph_simple(tree_nodes)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate layout
    pos = compute_layout_positions_simple(G, tree_nodes)
    print(f"Calculated positions for {len(pos)} nodes")
    
    # Get colors and sizes
    colors, sizes = determine_node_appearance_simple(tree_nodes)
    
    # Create labels
    labels = generate_simple_node_labels(tree_nodes)
    
    # Create the plot
    print("Creating simplified tree visualization...")
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    
    # Draw edges with clean styling
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=12, arrowstyle='->', alpha=0.6, width=1.0)
    
    # Draw nodes
    node_list = sorted(tree_nodes.keys())
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.8, linewidths=1.5, edgecolors='black')
    
    # Draw stars for optimal nodes (0% gap)
    optimal_nodes = [node_id for node_id, node_data in tree_nodes.items() 
                    if node_data.is_optimal]
    
    if optimal_nodes:
        optimal_positions = [pos[node_id] for node_id in optimal_nodes]
        optimal_x = [pos[0] for pos in optimal_positions]
        optimal_y = [pos[1] for pos in optimal_positions]
        
        # Draw gold stars on optimal nodes
        plt.scatter(optimal_x, optimal_y, s=120, c='gold', marker='*', 
                   edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
        
        print(f"Found {len(optimal_nodes)} optimal nodes: {optimal_nodes}")
    
    # Draw simple labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                          font_color='darkblue')
    
    # Create simplified legend
    legend_elements = create_simple_legend_elements()
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Count termination reasons for summary
    termination_counts = defaultdict(int)
    for node in tree_nodes.values():
        termination_counts[node.termination_reason] += 1
    
    # Add summary text
    summary_text = f"Nodes: {len(tree_nodes)}\n"
    summary_text += f"Active: {termination_counts.get('active', 0)}\n"
    summary_text += f"Bound Pruned: {termination_counts.get('bound_pruned', 0)}\n"
    summary_text += f"Infeasible: {termination_counts.get('infeasible', 0)}\n"
    summary_text += f"Integer Solutions: {termination_counts.get('integer_solution', 0)}"
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9),
            family='monospace')
    
    # Set title
    plt.title(f"Branch-and-Bound Tree Structure: {instance_name}\n"
             f"Node Size ∝ Subtree Count, Color = Termination Reason", 
             fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"tree_simple_{instance_name.replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, filename)
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved visualization to: {output_path}")
    
    # Show plot
    if show_plot:
        print("Displaying simplified tree plot...")
        plt.show()
    else:
        plt.close()
    
    # Return summary statistics
    return {
        'instance_name': instance_name,
        'total_nodes': len(tree_nodes),
        'optimal_nodes': len(optimal_nodes),
        'termination_counts': dict(termination_counts)
    }


def create_simple_legend_elements():
    """Create legend elements for the simplified tree visualization."""
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Active (Branched)'),
        mpatches.Patch(color='lightgreen', label='Bound Pruned'),
        mpatches.Patch(color='lightcoral', label='Infeasible'),
        mpatches.Patch(color='gold', label='Integer Solution'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=10, label='Optimal Node')
    ]
    return legend_elements


def process_multiple_tree_files_simple(input_path, output_dir=None, show_plots=False):
    """
    Process multiple structured tree files for simplified visualization.
    
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
        stats = create_simple_tree_plot(str(input_path), output_dir, show_plots)
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
            stats = create_simple_tree_plot(str(file_path), output_dir, False)
            results.append(stats)
        
        print(f"\nProcessed {len(tree_files)} files")
        if output_dir:
            print(f"All simplified visualizations saved to: {output_dir}")
    else:
        print(f"Invalid path: {input_path}")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Create simplified tree structure visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 visualize_tree_simple.py structured_tree_file.txt
  python3 visualize_tree_simple.py structured_trees/ --output-dir plots/
  python3 visualize_tree_simple.py structured_tree_file.txt --no-show
        """
    )
    
    parser.add_argument('input_path', help='Path to structured tree file or directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for saving plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    # Process the files
    results = process_multiple_tree_files_simple(
        input_path=args.input_path,
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )
    
    # Print overall summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY - SIMPLIFIED TREE VISUALIZATION")
        print(f"{'='*60}")
        print(f"Processed {len(results)} tree files")
        
        total_nodes = sum(r.get('total_nodes', 0) for r in results)
        total_optimal = sum(r.get('optimal_nodes', 0) for r in results)
        
        print(f"Total nodes across all trees: {total_nodes}")
        print(f"Total optimal nodes: {total_optimal}")


if __name__ == "__main__":
    main()