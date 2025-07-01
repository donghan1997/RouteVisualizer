#!/usr/bin/env python3
"""
Simplified Branch-and-Bound Tree Analysis Pipeline

This script provides a streamlined pipeline focusing on tree structure visualization
without dual gain estimation details. Shows:
- Clean tree structure
- Node termination reasons (bound pruned, infeasible, integer solution)
- Smaller node sizes for better readability

Usage:
    python3 tree_analysis_pipeline_simple.py <log_file.out> [--output-dir <dir>] [--save-intermediates]

Author: Heinrich (Simplified)
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

# Import functions from the core modules
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
    from core.visualize_tree_simple import (
        determine_node_appearance_simple,
        generate_simple_node_labels,
        create_simple_legend_elements,
        determine_termination_reason
    )
    # Use the original load_structured_tree function from the full visualize_tree
    from core.visualize_tree import (
        load_structured_tree,
        build_networkx_graph,
        compute_layout_positions
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure the following files are available:")
    print("- core/extract_tree_data.py")
    print("- core/build_tree_structure.py") 
    print("- visualize_tree_simple.py")
    sys.exit(1)


# Configuration constants
DEFAULT_FIGURE_SIZE = (14, 10)
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
        """Extract tree data from solver log file."""
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
    """Stage 2: Build structured tree with parent-child relationships."""
    
    def __init__(self):
        super().__init__("Building structured tree (simplified)", 2)
    
    def build(self, tree_detail_path, output_dir=None):
        """Build structured tree with parent-child relationships."""
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


class SimpleTreeVisualizer(PipelineStage):
    """Stage 3: Create simplified tree visualization in matplotlib subplot."""
    
    def __init__(self):
        super().__init__("Creating simplified tree visualization", 3)
    
    def visualize_in_subplot(self, structured_tree_path, ax, title_suffix=""):
        """Create simplified tree visualization in provided matplotlib axis."""
        self.print_header()
        
        # Parse the structured tree file using the original function
        tree_nodes, instance_name = load_structured_tree(structured_tree_path)
        
        if not tree_nodes:
            raise ValueError(f"No nodes found in {structured_tree_path}")
        
        print(f"Visualizing simplified tree for {instance_name} with {len(tree_nodes)} nodes")
        
        # Add termination reason to each node
        for node_id, node_data in tree_nodes.items():
            # We need to determine termination reason from the structured tree data
            children_ids = getattr(node_data, 'children_ids', [])
            
            # Determine termination reason based on existing node data
            if children_ids:
                termination_reason = 'active'
            else:
                # For leaf nodes, we need to make educated guesses
                # This is a simplified heuristic - could be improved with actual LB/UB values
                gap_percent = getattr(node_data, 'gap_percent', None)
                if gap_percent is not None:
                    if abs(gap_percent) < 0.001:
                        termination_reason = 'integer_solution'
                    elif gap_percent < -50:  # Very negative gap might indicate infeasible
                        termination_reason = 'infeasible'
                    else:
                        termination_reason = 'bound_pruned'
                else:
                    termination_reason = 'bound_pruned'  # Default for leaf nodes
            
            node_data.termination_reason = termination_reason
            node_data.is_optimal = hasattr(node_data, 'gap_percent') and node_data.gap_percent is not None and abs(node_data.gap_percent) < 0.001
        
        # Create graph using original function
        G = build_networkx_graph(tree_nodes)
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate layout using original function
        pos = compute_layout_positions(G, tree_nodes)
        print(f"Calculated positions for {len(pos)} nodes")
        
        # Get colors and sizes with simplified appearance
        colors, sizes = determine_node_appearance_simple(tree_nodes)
        
        # Create simplified labels
        labels = generate_simple_node_labels(tree_nodes)
        
        # Draw the tree on the provided axis
        print("Drawing simplified tree visualization...")
        
        # Draw edges with clean styling
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=12, arrowstyle='->', alpha=0.6, width=1.0, ax=ax)
        
        # Draw nodes
        node_list = sorted(tree_nodes.keys())
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                              node_color=colors, node_size=sizes, 
                              alpha=0.8, linewidths=1.5, edgecolors='black', ax=ax)
        
        # Draw stars for optimal nodes (0% gap)
        optimal_nodes = [node_id for node_id, node_data in tree_nodes.items() 
                        if getattr(node_data, 'is_optimal', False)]
        
        if optimal_nodes:
            optimal_positions = [pos[node_id] for node_id in optimal_nodes]
            optimal_x = [pos[0] for pos in optimal_positions]
            optimal_y = [pos[1] for pos in optimal_positions]
            
            ax.scatter(optimal_x, optimal_y, s=120, c='gold', marker='*', 
                      edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
            
            print(f"Found {len(optimal_nodes)} optimal nodes: {optimal_nodes}")
        
        # Draw simple labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                              font_color='darkblue', ax=ax)
        
        # Count termination reasons for summary
        termination_counts = defaultdict(int)
        for node in tree_nodes.values():
            termination_counts[getattr(node, 'termination_reason', 'unknown')] += 1
        
        # Add summary text in top-left corner
        summary_text = f"Nodes: {len(tree_nodes)}\n"
        summary_text += f"Active: {termination_counts.get('active', 0)}\n"
        summary_text += f"Bound Pruned: {termination_counts.get('bound_pruned', 0)}\n"
        summary_text += f"Infeasible: {termination_counts.get('infeasible', 0)}\n"
        summary_text += f"Integer Solutions: {termination_counts.get('integer_solution', 0)}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9),
                family='monospace')
        
        # Set title
        full_title = f"B&B Tree Structure: {instance_name}"
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
            'termination_counts': dict(termination_counts)
        }


class SimpleTreeAnalysisPipeline:
    """Complete simplified branch-and-bound tree analysis pipeline."""
    
    def __init__(self):
        self.extractor = TreeDataExtractor()
        self.builder = TreeStructureBuilder()
        self.visualizer = SimpleTreeVisualizer()
    
    def run_pipeline(self, log_file_path, output_dir=None, save_intermediates=False, 
                    show_plot=True, debug=False):
        """
        Run the complete simplified branch-and-bound tree analysis pipeline.
        
        Args:
            log_file_path (str): Path to the solver log file (.out)
            output_dir (str): Directory to save output files (optional)
            save_intermediates (bool): Whether to save intermediate files permanently
            show_plot (bool): Whether to display the plot
            debug (bool): Enable debug output for Stage 1
            
        Returns:
            dict: Summary results from all stages
        """
        print(f"Starting simplified B&B tree analysis pipeline for: {log_file_path}")
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
            temp_dir = tempfile.mkdtemp(prefix="bb_pipeline_simple_")
            work_dir = temp_dir
        
        try:
            # Stage 1: Extract tree data
            tree_detail_path, instance_name, tree_sections = self.extractor.extract(
                log_file_path, work_dir, debug)
            
            # Stage 2: Build structured tree
            structured_tree_path, level_sorted_nodes = self.builder.build(
                tree_detail_path, work_dir)
            
            # Stage 3: Create simplified visualization
            print("Creating simplified matplotlib figure...")
            fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGURE_SIZE)
            
            tree_stats = self.visualizer.visualize_in_subplot(structured_tree_path, ax)
            
            # Add legend
            legend_elements = create_simple_legend_elements()
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
            
            # Add overall title and description
            fig.suptitle(f"Simplified Branch-and-Bound Tree Analysis\n"
                        f"Input: {os.path.basename(log_file_path)}", 
                        fontsize=14, fontweight='bold')
            
            # Add description text
            description = ("Node Size ∝ Subtree Count, Color = Termination Reason\n"
                          "Focus: Tree Structure without Dual Gain Estimation Details")
            fig.text(0.5, 0.02, description, ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            
            # Save plot if output directory specified
            if output_dir and save_intermediates:
                plot_filename = f"bb_tree_simple_{instance_name}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                print(f"Saving simplified plot to: {plot_path}")
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
            self._print_pipeline_summary(tree_stats, instance_name)
            
            # Prepare results summary
            results = {
                'instance_name': instance_name,
                'log_file': log_file_path,
                'tree_detail_file': tree_detail_path if save_intermediates else None,
                'structured_tree_file': structured_tree_path if save_intermediates else None,
                'tree_stats': tree_stats,
                'success': True
            }
            
            print(f"\nSimplified pipeline completed successfully!")
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
    
    def _print_pipeline_summary(self, tree_stats, instance_name):
        """Print simplified pipeline summary statistics."""
        print("\n" + "=" * 80)
        print("SIMPLIFIED PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Instance: {instance_name}")
        print(f"Total nodes: {tree_stats.get('total_nodes', 0)}")
        print(f"Optimal nodes: {tree_stats.get('optimal_nodes', 0)}")
        
        termination_counts = tree_stats.get('termination_counts', {})
        print("\nNode Termination Summary:")
        print(f"  Active (branched): {termination_counts.get('active', 0)}")
        print(f"  Bound pruned: {termination_counts.get('bound_pruned', 0)}")
        print(f"  Infeasible: {termination_counts.get('infeasible', 0)}")
        print(f"  Integer solutions: {termination_counts.get('integer_solution', 0)}")
        
        # Calculate percentages
        total = tree_stats.get('total_nodes', 0)
        if total > 0:
            print("\nNode Termination Percentages:")
            for reason, count in termination_counts.items():
                percentage = (count / total) * 100
                print(f"  {reason.replace('_', ' ').title()}: {percentage:.1f}%")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Simplified Branch-and-Bound Tree Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This simplified version focuses on tree structure visualization without 
dual gain estimation details. Shows:
- Clean tree structure
- Node termination reasons (bound pruned, infeasible, integer solution) 
- Smaller node sizes for better readability

Examples:
  python3 tree_analysis_pipeline_simple.py solver_output.out
  python3 tree_analysis_pipeline_simple.py solver_output.out --output-dir results/ --save-intermediates
  python3 tree_analysis_pipeline_simple.py solver_output.out --no-show --debug
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
    
    # Create and run the simplified pipeline
    pipeline = SimpleTreeAnalysisPipeline()
    results = pipeline.run_pipeline(
        log_file_path=args.log_file,
        output_dir=args.output_dir,
        save_intermediates=args.save_intermediates,
        show_plot=not args.no_show,
        debug=args.debug
    )
    
    # Exit with appropriate code
    if results['success']:
        print("\nSimplified pipeline execution completed successfully!")
        sys.exit(0)
    else:
        print(f"\nSimplified pipeline execution failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()