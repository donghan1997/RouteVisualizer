#!/usr/bin/env python3
"""
Branch-and-Bound Tree Analysis Pipeline

This script provides a complete end-to-end pipeline for analyzing branch-and-bound solver logs.
It combines the functionality of three separate modules into a single streamlined process:

1. Stage 1: Extract tree data from solver logs
2. Stage 2: Build structured tree with size estimations  
3. Stage 3: Visualize the tree structure in a matplotlib subplot

Usage:
    python3 tree_analysis_pipeline.py <log_file.out> [--output-dir <dir>] [--save-intermediates]
    python3 tree_analysis_pipeline.py raw_logs/output_2bwoeb_CVRP_120_1.txt --output-dir results/ --save-intermediates

Author: Heinrich (Refined)
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

# Import matplotlib with fallback installation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install required packages:")
    print("pip install matplotlib networkx")
    sys.exit(1)

from collections import defaultdict

# Import functions from the refined core modules
try:
    from core.extract_tree_data import (
        extract_tree_sections,
        get_instance_name,
        create_output_filename,
        format_tree_report
    )
    from core.build_tree_structure import (
        load_tree_data,
        create_tree_hierarchy,
        sort_nodes_by_level,
        create_structured_report
    )
    from core.visualize_tree import (
        load_structured_tree,
        build_networkx_graph,
        compute_layout_positions,
        determine_node_appearance,
        generate_node_labels,
        add_level_stats,
        calculate_estimation_quality,
        print_tree_statistics
    )
except ImportError as e:
    print(f"Error importing required core modules: {e}")
    print("Please ensure the following files are in the core/ directory:")
    print("- core/extract_tree_data.py")
    print("- core/build_tree_structure.py") 
    print("- core/visualize_tree.py")
    print("\nOr update the import paths to match your project structure.")
    sys.exit(1)


# Configuration constants
GOOD_ESTIMATE_RANGE = (0.5, 2.0)
DEFAULT_FIGURE_SIZE = (16, 10)
DEFAULT_DPI = 300


class PipelineStage:
    """Base class for pipeline stages with common functionality."""
    
    def __init__(self, stage_name, stage_number):
        self.stage_name = stage_name
        self.stage_number = stage_number
    
    def print_header(self):
        """Print stage header."""
        print(f"=== STAGE {self.stage_number}: {self.stage_name} ===")
    
    def print_completion(self):
        """Print stage completion message."""
        print(f"Stage {self.stage_number} completed successfully\n")


class TreeDataExtractor(PipelineStage):
    """Stage 1: Extract tree data from solver log files."""
    
    def __init__(self):
        super().__init__("Extracting tree data from solver log", 1)
    
    def extract(self, log_file_path, output_dir=None, debug=False):
        """
        Extract tree data from solver log file.
        
        Args:
            log_file_path (str): Path to the solver log (.out) file
            output_dir (str): Directory to save tree detail file (optional)
            debug (bool): Enable debug output
            
        Returns:
            tuple: (tree_detail_file_path, instance_name, tree_sections)
        """
        self.print_header()
        print(f"Processing: {os.path.basename(log_file_path)}")
        
        # Extract instance name from the log file
        instance_name = get_instance_name(log_file_path)
        print(f"Instance name: {instance_name}")
        
        # Parse tree data from the log file
        tree_sections = extract_tree_sections(log_file_path, debug=debug)
        
        if not tree_sections:
            raise ValueError(f"No tree data found in {log_file_path}")
        
        print(f"Found {len(tree_sections)} tree sections")
        
        # Generate output filename and path
        tree_filename = create_output_filename(instance_name)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tree_detail_path = os.path.join(output_dir, tree_filename)
        else:
            # Use temporary directory if no output specified
            temp_dir = tempfile.mkdtemp(prefix="bb_tree_")
            tree_detail_path = os.path.join(temp_dir, tree_filename)
        
        # Generate and save tree detail output
        tree_output = format_tree_report(tree_sections, instance_name)
        
        with open(tree_detail_path, 'w', encoding='utf-8') as f:
            f.write(tree_output)
        
        print(f"Tree detail file saved: {tree_detail_path}")
        self.print_completion()
        
        return tree_detail_path, instance_name, tree_sections


class TreeStructureBuilder(PipelineStage):
    """Stage 2: Build structured tree with parent-child relationships and size estimations."""
    
    def __init__(self):
        super().__init__("Building structured tree with estimations", 2)
    
    def build(self, tree_detail_path, output_dir=None):
        """
        Build structured tree with parent-child relationships and size estimations.
        
        Args:
            tree_detail_path (str): Path to tree detail file from Stage 1
            output_dir (str): Directory to save structured tree file (optional)
            
        Returns:
            tuple: (structured_tree_path, level_sorted_nodes)
        """
        self.print_header()
        print(f"Processing: {os.path.basename(tree_detail_path)}")
        
        # Parse tree data file
        tree_nodes = load_tree_data(tree_detail_path)
        
        if not tree_nodes:
            raise ValueError(f"No nodes found in {tree_detail_path}")
        
        print(f"Parsed {len(tree_nodes)} nodes from tree detail file")
        
        # Build tree structure
        root = create_tree_hierarchy(tree_nodes)
        
        if not root:
            raise ValueError(f"Could not build tree structure from {tree_detail_path}")
        
        print(f"Built tree structure with root node {root.idx}")
        
        # Arrange nodes by level
        level_sorted_nodes = sort_nodes_by_level(root)
        print(f"Arranged {len(level_sorted_nodes)} nodes by tree level")
        
        # Extract instance name for output filename
        instance_name = os.path.splitext(os.path.basename(tree_detail_path))[0]
        # Remove 'tree_' prefix if present
        if instance_name.startswith('tree_'):
            instance_name = instance_name[5:]
        
        # Generate structured tree output
        structured_output = create_structured_report(level_sorted_nodes, instance_name)
        
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
        self.print_completion()
        
        return structured_tree_path, level_sorted_nodes


class TreeVisualizer(PipelineStage):
    """Stage 3: Create tree visualization in matplotlib subplot."""
    
    def __init__(self):
        super().__init__("Creating tree visualization", 3)
    
    def visualize_in_subplot(self, structured_tree_path, ax, title_suffix=""):
        """
        Create tree visualization in provided matplotlib axis.
        
        Args:
            structured_tree_path (str): Path to structured tree file from Stage 2
            ax (matplotlib.Axes): Matplotlib axis to draw the tree on
            title_suffix (str): Additional text for plot title
            
        Returns:
            dict: Summary statistics about the tree
        """
        self.print_header()
        
        # Parse the structured tree file
        tree_nodes, instance_name = load_structured_tree(structured_tree_path)
        
        if not tree_nodes:
            raise ValueError(f"No nodes found in {structured_tree_path}")
        
        print(f"Visualizing tree for {instance_name} with {len(tree_nodes)} nodes")
        
        # Create graph
        G = build_networkx_graph(tree_nodes)
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate layout
        pos = compute_layout_positions(G, tree_nodes)
        print(f"Calculated positions for {len(pos)} nodes")
        
        # Get colors and sizes
        colors, sizes = determine_node_appearance(tree_nodes)
        
        # Create labels
        labels = generate_node_labels(tree_nodes)
        
        # Draw the tree on the provided axis
        print("Drawing tree visualization...")
        
        # Draw edges with better styling
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=15, arrowstyle='->', alpha=0.7, width=1.5,
                              connectionstyle="arc3,rad=0.1", ax=ax)
        
        # Draw nodes
        node_list = sorted(tree_nodes.keys())
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                              node_color=colors, node_size=sizes, 
                              alpha=0.9, linewidths=2, edgecolors='black', ax=ax)
        
        # Draw stars for optimal nodes (0% gap)
        optimal_nodes = [node_id for node_id, node_data in tree_nodes.items() 
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
        quality_stats = calculate_estimation_quality(tree_nodes)
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
        
        # Add level statistics
        add_level_stats(tree_nodes, pos, ax)
        
        # Set title
        full_title = f"B&B Tree: {instance_name}"
        if title_suffix:
            full_title += f" {title_suffix}"
        
        ax.set_title(full_title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        self.print_completion()
        
        # Return summary statistics
        return {
            'instance_name': instance_name,
            'total_nodes': len(tree_nodes),
            'optimal_nodes': len(optimal_nodes),
            'quality_stats': quality_stats,
            'level_stats': self._calculate_level_statistics(tree_nodes)
        }
    
    def _calculate_level_statistics(self, tree_nodes):
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


def create_legend_elements():
    """Create legend elements for the tree visualization."""
    legend_elements = [
        mpatches.Patch(color='lightgreen', label=f'Good Estimate ({GOOD_ESTIMATE_RANGE[0]}x ≤ Est/Real ≤ {GOOD_ESTIMATE_RANGE[1]}x)'),
        mpatches.Patch(color='orange', label=f'Over-estimated (Est/Real > {GOOD_ESTIMATE_RANGE[1]}x)'),
        mpatches.Patch(color='lightblue', label=f'Under-estimated (Est/Real < {GOOD_ESTIMATE_RANGE[0]}x)'),
        mpatches.Patch(color='red', label='Infinite Estimate'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=10, label='Optimal Node (0% gap)')
    ]
    return legend_elements


class TreeAnalysisPipeline:
    """Complete branch-and-bound tree analysis pipeline."""
    
    def __init__(self):
        self.extractor = TreeDataExtractor()
        self.builder = TreeStructureBuilder()
        self.visualizer = TreeVisualizer()
    
    def run_pipeline(self, log_file_path, output_dir=None, save_intermediates=False, 
                    show_plot=True, debug=False):
        """
        Run the complete branch-and-bound tree analysis pipeline.
        
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
            tree_detail_path, instance_name, tree_sections = self.extractor.extract(
                log_file_path, work_dir, debug)
            
            # Stage 2: Build structured tree
            structured_tree_path, level_sorted_nodes = self.builder.build(
                tree_detail_path, work_dir)
            
            # Stage 3: Create visualization
            print("Creating matplotlib figure...")
            fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGURE_SIZE)
            
            tree_stats = self.visualizer.visualize_in_subplot(structured_tree_path, ax)
            
            # Add legend
            legend_elements = create_legend_elements()
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
                plt.savefig(plot_path, dpi=DEFAULT_DPI, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"Plot saved successfully")
            
            # Show plot
            if show_plot:
                print("Displaying interactive plot...")
                plt.show()
            else:
                plt.close()
            
            # Print console summary
            self._print_pipeline_summary(level_sorted_nodes, instance_name)
            
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
    
    def _print_pipeline_summary(self, level_sorted_nodes, instance_name):
        """Print pipeline summary statistics."""
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        
        # Convert BranchNode objects to compatible format for print_tree_statistics
        compatible_nodes = {}
        for node in level_sorted_nodes:
            # Create a compatible node object
            class CompatibleNode:
                def __init__(self, branch_node):
                    self.node_id = branch_node.idx  # BranchNode uses .idx
                    self.level = branch_node.level
                    self.subtree_count = branch_node.subtree_count
                    self.estimated_subtree_size = branch_node.estimated_size
                    self.gap_percent = getattr(branch_node.data, 'gap', None)
            
            compatible_nodes[node.idx] = CompatibleNode(node)
        
        print_tree_statistics(compatible_nodes, instance_name)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Complete Branch-and-Bound Tree Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tree_analysis_pipeline.py solver_output.out
  python3 tree_analysis_pipeline.py solver_output.out --output-dir results/ --save-intermediates
  python3 tree_analysis_pipeline.py solver_output.out --no-show --debug
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
    
    # Create and run the pipeline
    pipeline = TreeAnalysisPipeline()
    results = pipeline.run_pipeline(
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