#!/usr/bin/env python3
"""
Branch-and-Bound Tree Analysis Pipeline

This script provides a complete end-to-end pipeline for analyzing branch-and-bound solver logs.
It combines the functionality of three separate scripts into a single streamlined process:

1. Stage 1: Extract tree data from solver logs
2. Stage 2: Build structured tree with size estimations  
3. Stage 3: Visualize the tree structure in a matplotlib subplot

Usage:
    python3 bb_tree_pipeline.py <log_file.out> [--output-dir <dir>] [--save-intermediates]

Author: Heinrich
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# Import functions from the three original scripts
# Note: Make sure these files are in the same directory or in Python path
try:
    from generate_tree_details import (
        parse_out_file_for_tree,
        extract_instance_name_from_out_file,
        generate_tree_filename,
        generate_tree_output
    )
    from arrange_tree_structure_selflog_distinctnode_kmax_simpler import (
        parse_tree_data_file,
        build_tree_structure,
        arrange_nodes_by_level,
        generate_simplified_tree_output
    )
    from plot_tree_structure_withlogs_exlevel import (
        parse_structured_tree_file,
        create_tree_graph,
        calculate_node_positions,
        get_node_colors_and_sizes,
        create_node_labels,
        add_level_statistics_to_plot,
        calculate_estimation_quality,
        print_tree_statistics
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure the following files are in the same directory:")
    print("- generate_tree_details.py")
    print("- arrange_tree_structure_selflog_distinctnode_kmax_simpler.py") 
    print("- plot_tree_structure_withlogs_exlevel.py")
    sys.exit(1)

import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict


def stage1_extract_tree_data(log_file_path, output_dir=None, debug=False):
    """
    Stage 1: Extract tree data from solver log file
    
    Args:
        log_file_path (str): Path to the solver log (.out) file
        output_dir (str): Directory to save tree detail file (optional)
        debug (bool): Enable debug output
        
    Returns:
        tuple: (tree_detail_file_path, instance_name, tree_data)
    """
    print(f"=== STAGE 1: Extracting tree data from {os.path.basename(log_file_path)} ===")
    
    # Extract instance name from the log file
    instance_name = extract_instance_name_from_out_file(log_file_path)
    print(f"Instance name: {instance_name}")
    
    # Parse tree data from the log file
    tree_data = parse_out_file_for_tree(log_file_path, debug=debug)
    
    if not tree_data:
        raise ValueError(f"No tree data found in {log_file_path}")
    
    print(f"Found {len(tree_data)} tree sections")
    
    # Generate output filename and path
    tree_filename = generate_tree_filename(instance_name)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tree_detail_path = os.path.join(output_dir, tree_filename)
    else:
        # Use temporary directory if no output specified
        temp_dir = tempfile.mkdtemp(prefix="bb_tree_")
        tree_detail_path = os.path.join(temp_dir, tree_filename)
    
    # Generate and save tree detail output
    tree_output = generate_tree_output(tree_data, instance_name)
    
    with open(tree_detail_path, 'w', encoding='utf-8') as f:
        f.write(tree_output)
    
    print(f"Tree detail file saved: {tree_detail_path}")
    print(f"Stage 1 completed successfully\n")
    
    return tree_detail_path, instance_name, tree_data


def stage2_build_structured_tree(tree_detail_path, output_dir=None):
    """
    Stage 2: Build structured tree with parent-child relationships and size estimations
    
    Args:
        tree_detail_path (str): Path to tree detail file from Stage 1
        output_dir (str): Directory to save structured tree file (optional)
        
    Returns:
        tuple: (structured_tree_path, arranged_nodes)
    """
    print(f"=== STAGE 2: Building structured tree from {os.path.basename(tree_detail_path)} ===")
    
    # Parse tree data file
    nodes = parse_tree_data_file(tree_detail_path)
    
    if not nodes:
        raise ValueError(f"No nodes found in {tree_detail_path}")
    
    print(f"Parsed {len(nodes)} nodes from tree detail file")
    
    # Build tree structure
    root = build_tree_structure(nodes)
    
    if not root:
        raise ValueError(f"Could not build tree structure from {tree_detail_path}")
    
    print(f"Built tree structure with root node {root.idx}")
    
    # Arrange nodes by level
    arranged_nodes = arrange_nodes_by_level(root)
    print(f"Arranged {len(arranged_nodes)} nodes by tree level")
    
    # Extract instance name for output filename
    instance_name = os.path.splitext(os.path.basename(tree_detail_path))[0]
    # Remove 'tree_' prefix if present
    if instance_name.startswith('tree_'):
        instance_name = instance_name[5:]
    
    # Generate structured tree output
    structured_output = generate_simplified_tree_output(arranged_nodes, instance_name)
    
    # Determine output path
    structured_filename = f"structured_tree_{instance_name}.txt"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        structured_tree_path = os.path.join(output_dir, structured_filename)
    else:
        # Use same directory as tree detail file
        structured_tree_path = os.path.join(os.path.dirname(tree_detail_path), structured_filename)
    
    # Save structured tree file
    with open(structured_tree_path, 'w', encoding='utf-8') as f:
        f.write(structured_output)
    
    print(f"Structured tree file saved: {structured_tree_path}")
    print(f"Stage 2 completed successfully\n")
    
    return structured_tree_path, arranged_nodes


def stage3_visualize_tree_in_subplot(structured_tree_path, ax, title_suffix=""):
    """
    Stage 3: Create tree visualization in provided matplotlib axis
    
    Args:
        structured_tree_path (str): Path to structured tree file from Stage 2
        ax (matplotlib.Axes): Matplotlib axis to draw the tree on
        title_suffix (str): Additional text for plot title
        
    Returns:
        dict: Summary statistics about the tree
    """
    print(f"=== STAGE 3: Creating tree visualization ===")
    
    # Parse the structured tree file
    nodes, instance_name = parse_structured_tree_file(structured_tree_path)
    
    if not nodes:
        raise ValueError(f"No nodes found in {structured_tree_path}")
    
    print(f"Visualizing tree for {instance_name} with {len(nodes)} nodes")
    
    # Create graph
    G = create_tree_graph(nodes)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate layout
    pos = calculate_node_positions(G, nodes)
    print(f"Calculated positions for {len(pos)} nodes")
    
    # Get colors and sizes
    colors, sizes = get_node_colors_and_sizes(nodes)
    
    # Create labels
    labels = create_node_labels(nodes)
    
    # Draw the tree on the provided axis
    print("Drawing tree visualization...")
    
    # Draw edges with better styling
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', alpha=0.7, width=1.5,
                          connectionstyle="arc3,rad=0.1", ax=ax)
    
    # Draw nodes
    node_list = sorted(nodes.keys())
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.9, linewidths=2, edgecolors='black', ax=ax)
    
    # Draw stars for optimal nodes (0% gap)
    optimal_nodes = [node_id for node_id, node_data in nodes.items() 
                    if hasattr(node_data, 'gap_percent') and node_data.gap_percent is not None and abs(node_data.gap_percent) < 0.001]
    
    if optimal_nodes:
        optimal_positions = [pos[node_id] for node_id in optimal_nodes]
        optimal_x = [pos[0] for pos in optimal_positions]
        optimal_y = [pos[1] for pos in optimal_positions]
        
        ax.scatter(optimal_x, optimal_y, s=150, c='gold', marker='*', 
                  edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
        
        print(f"Found {len(optimal_nodes)} optimal nodes (0% gap): {optimal_nodes}")
    
    # Draw labels with offset positioning
    label_pos = {node: (x - 0.3, y) for node, (x, y) in pos.items()}
    
    # Draw labels manually to work with subplot
    for node_id, (x, y) in label_pos.items():
        if labels.get(node_id, ""):  # Only draw non-empty labels
            ax.text(x, y, labels[node_id], fontsize=9, fontweight='bold',
                   color='darkblue', verticalalignment='center',
                   horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Calculate and add estimation quality
    quality_stats = calculate_estimation_quality(nodes)
    all_stats = quality_stats['all']
    excl_stats = quality_stats['excluding_real_1']
    
    # Create quality text
    if all_stats['finite'] > 0:
        quality_text = "Estimation Quality:\n"
        quality_text += f"Good: {all_stats['good']} ({all_stats['good']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Over: {all_stats['over']} ({all_stats['over']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Under: {all_stats['under']} ({all_stats['under']/all_stats['finite']*100:.1f}%)"
        
        if excl_stats['finite'] > 0:
            quality_text += f"\n\nExcl. real=1:\n"
            quality_text += f"Good: {excl_stats['good']} ({excl_stats['good']/excl_stats['finite']*100:.1f}%)"
        
        # Add text box in top-right corner
        ax.text(0.98, 0.98, quality_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
                family='monospace')
    
    # Add level statistics - use custom function to avoid import issues
    level_stats = calculate_level_statistics_for_stage3(nodes)
    
    if level_stats:
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
                level_y = level_positions[0][1]
                
                # Create statistics text
                if stats['inf'] > 0:
                    stats_text = f"L{level}: {good_percent:.1f}%\n({stats['good']}/{finite_total}, {stats['inf']} inf)"
                else:
                    stats_text = f"L{level}: {good_percent:.1f}%\n({stats['good']}/{finite_total})"
                
                # Position text to the right of the level
                text_x = max_x + 2.0
                ax.text(text_x, level_y, stats_text, 
                       fontsize=8, fontweight='bold',
                       verticalalignment='center', horizontalalignment='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                       family='monospace')
    
    # Set title
    full_title = f"B&B Tree: {instance_name}"
    if title_suffix:
        full_title += f" {title_suffix}"
    
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    print(f"Stage 3 completed successfully")
    
    # Return summary statistics
    return {
        'instance_name': instance_name,
        'total_nodes': len(nodes),
        'optimal_nodes': len(optimal_nodes),
        'quality_stats': quality_stats,
        'level_stats': level_stats
    }


def calculate_level_statistics_for_stage3(nodes):
    """Calculate estimation quality statistics for each level - Stage 3 compatible version."""
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


def create_legend_for_subplot():
    """Create legend elements for the tree visualization."""
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Good Estimate (0.5x ≤ Est/Real ≤ 2x)'),
        mpatches.Patch(color='orange', label='Over-estimated (Est/Real > 2x)'),
        mpatches.Patch(color='lightblue', label='Under-estimated (Est/Real < 0.5x)'),
        mpatches.Patch(color='red', label='Infinite Estimate'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=10, label='Optimal Node (0% gap)')
    ]
    return legend_elements


def run_complete_pipeline(log_file_path, output_dir=None, save_intermediates=False, 
                         show_plot=True, debug=False):
    """
    Run the complete branch-and-bound tree analysis pipeline
    
    Args:
        log_file_path (str): Path to the solver log file (.out)
        output_dir (str): Directory to save output files (optional)
        save_intermediates (bool): Whether to save intermediate files permanently
        show_plot (bool): Whether to display the plot
        debug (bool): Enable debug output for Stage 1
        
    Returns:
        dict: Summary results from all stages
    """
    print(f"Starting complete B&B tree analysis pipeline for: {log_file_path}")
    print("=" * 80)
    
    # Validate input file
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    # Setup output directory
    temp_dir = None
    if save_intermediates:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            work_dir = output_dir
        else:
            work_dir = os.path.dirname(log_file_path)
    else:
        # Use temporary directory
        temp_dir = tempfile.mkdtemp(prefix="bb_pipeline_")
        work_dir = temp_dir
    
    try:
        # Stage 1: Extract tree data
        tree_detail_path, instance_name, tree_data = stage1_extract_tree_data(
            log_file_path, work_dir, debug)
        
        # Stage 2: Build structured tree
        structured_tree_path, arranged_nodes = stage2_build_structured_tree(
            tree_detail_path, work_dir)
        
        # Stage 3: Create visualization
        print("Creating matplotlib figure...")
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        tree_stats = stage3_visualize_tree_in_subplot(structured_tree_path, ax)
        
        # Add legend
        legend_elements = create_legend_for_subplot()
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
        
        # Add overall title and description
        fig.suptitle(f"Branch-and-Bound Tree Analysis Pipeline Results\n"
                    f"Input: {os.path.basename(log_file_path)}", 
                    fontsize=14, fontweight='bold')
        
        # Add description text
        description = ("Node Format: (Real, Est), Size ∝ Real Count, Color = Quality\n"
                      "Level Stats: % Good Estimates (excluding trivial real=1 nodes)")
        fig.text(0.5, 0.02, description, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # Save plot if output directory specified
        if output_dir and save_intermediates:
            plot_filename = f"bb_tree_analysis_{instance_name}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            print(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Plot saved successfully")
        
        # Show plot
        if show_plot:
            print("Displaying interactive plot...")
            plt.show()
        else:
            plt.close()
        
        # Print console summary
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        
        # Convert TreeNode objects to compatible format for print_tree_statistics
        # TreeNode objects have .idx, but print_tree_statistics expects objects with .node_id
        compatible_nodes = {}
        for node in arranged_nodes:
            # Create a compatible node object
            class CompatibleNode:
                def __init__(self, tree_node):
                    self.node_id = tree_node.idx  # TreeNode uses .idx
                    self.level = tree_node.level
                    self.subtree_count = tree_node.subtree_count
                    self.estimated_subtree_size = tree_node.estimated_size
                    self.gap_percent = getattr(tree_node.data, 'gap', None)
            
            compatible_nodes[node.idx] = CompatibleNode(node)
        
        print_tree_statistics(compatible_nodes, instance_name)
        
        # Prepare results summary
        results = {
            'instance_name': instance_name,
            'log_file': log_file_path,
            'tree_detail_file': tree_detail_path if save_intermediates else None,
            'structured_tree_file': structured_tree_path if save_intermediates else None,
            'tree_stats': tree_stats,
            'success': True
        }
        
        print(f"\nPipeline completed successfully!")
        if save_intermediates:
            print(f"Intermediate files saved in: {work_dir}")
        else:
            print("Intermediate files will be cleaned up automatically")
        
        return results
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return {
            'log_file': log_file_path,
            'error': str(e),
            'success': False
        }
        
    finally:
        # Clean up temporary directory if used
        if temp_dir and not save_intermediates:
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Complete Branch-and-Bound Tree Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bb_tree_pipeline.py solver_output.out
  python3 bb_tree_pipeline.py solver_output.out --output-dir results/ --save-intermediates
  python3 bb_tree_pipeline.py solver_output.out --no-show --debug
        """
    )
    
    parser.add_argument('log_file', 
                       help='Path to solver log file (.out)')
    parser.add_argument('--output-dir', '-o', 
                       help='Directory to save output files')
    parser.add_argument('--save-intermediates', '-s', action='store_true',
                       help='Save intermediate tree detail and structured tree files')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot interactively')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for tree extraction')
    
    args = parser.parse_args()
    
    # Run the complete pipeline
    results = run_complete_pipeline(
        log_file_path=args.log_file,
        output_dir=args.output_dir,
        save_intermediates=args.save_intermediates,
        show_plot=not args.no_show,
        debug=args.debug
    )
    
    # Exit with appropriate code
    if results['success']:
        print("\nPipeline execution completed successfully!")
        sys.exit(0)
    else:
        print(f"\nPipeline execution failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()