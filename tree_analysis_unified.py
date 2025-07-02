#!/usr/bin/env python3
"""
Unified Branch-and-Bound Tree Analysis with Streamlit Support

A modular approach with separate tree extraction and visualization functions,
enhanced with Streamlit logging and display capabilities:

1. getTree(log_file) - Extract and build tree structure 
2. plot_complex(tree) - Full visualization with estimation details
3. plot_simple(tree) - Clean visualization focusing on structure

Usage:
    # Command line / Jupyter
    tree = getTree("solver_output.log")
    plot_simple(tree)
    
    # Streamlit
    import streamlit as st
    tree = getTree("solver_output.log")
    plot_simple(tree, streamlit_container=st)
    
Author: Heinrich (Unified + Streamlit)
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
import io
import base64

# Try to import streamlit, fallback gracefully
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Import core modules
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
        print_tree_statistics,
        create_legend_elements
    )
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Please ensure core/ directory with required modules exists")
    sys.exit(1)


# Configuration constants
GOOD_ESTIMATE_RANGE = (0.5, 2.0)
DEFAULT_FIGURE_SIZE = (16, 10)
SIMPLE_FIGURE_SIZE = (14, 10)
DEFAULT_DPI = 300
DEFAULT_NODE_SIZE = 100
NODE_SIZE_SCALE = 200


def log_message(message, level="info", streamlit_container=None):
    """Unified logging that works with both console and Streamlit."""
    if streamlit_container and STREAMLIT_AVAILABLE:
        if level == "success":
            streamlit_container.success(message)
        elif level == "error":
            streamlit_container.error(message)
        elif level == "warning":
            streamlit_container.warning(message)
        else:
            streamlit_container.info(message)
    else:
        # Console output
        if level == "success":
            print(f"✅ {message}")
        elif level == "error":
            print(f"❌ {message}")
        elif level == "warning":
            print(f"⚠️ {message}")
        else:
            print(message)


class TreeData:
    """Container class for tree data and metadata."""
    
    def __init__(self, tree_nodes, instance_name, temp_files=None):
        self.tree_nodes = tree_nodes
        self.instance_name = instance_name
        self.temp_files = temp_files or []
        self._prepare_nodes()
    
    def _prepare_nodes(self):
        """Prepare nodes with additional attributes for visualization."""
        for node_id, node_data in self.tree_nodes.items():
            # Add termination reason for simple plotting
            children_ids = getattr(node_data, 'children_ids', [])
            
            if children_ids:
                termination_reason = 'active'
            else:
                # Heuristic for leaf nodes
                gap_percent = getattr(node_data, 'gap_percent', None)
                if gap_percent is not None:
                    if abs(gap_percent) < 0.001:
                        termination_reason = 'integer_solution'
                    elif gap_percent < -50:
                        termination_reason = 'infeasible'
                    else:
                        termination_reason = 'bound_pruned'
                else:
                    termination_reason = 'bound_pruned'
            
            node_data.termination_reason = termination_reason
            node_data.is_optimal = (hasattr(node_data, 'gap_percent') and 
                                  node_data.gap_percent is not None and 
                                  abs(node_data.gap_percent) < 0.001)
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
    
    def __del__(self):
        """Automatically clean up when object is destroyed."""
        self.cleanup()


def getTree(log_file, debug=False, keep_intermediates=False, streamlit_container=None):
    """
    Extract and build tree structure from solver log file.
    
    Args:
        log_file (str): Path to solver log file (.out)
        debug (bool): Enable debug output
        keep_intermediates (bool): Keep intermediate files (default: False)
        streamlit_container: Streamlit container for logging (optional)
        
    Returns:
        TreeData: Object containing tree nodes and metadata
        
    Example:
        # Console usage
        tree = getTree("solver_output.log")
        
        # Streamlit usage
        tree = getTree("solver_output.log", streamlit_container=st)
    """
    log_message(f"🌳 Extracting tree from: {os.path.basename(log_file)}", 
                streamlit_container=streamlit_container)
    
    # Validate input file
    if not os.path.exists(log_file):
        log_message(f"File not found: {log_file}", "error", streamlit_container)
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    temp_files = []
    
    try:
        # Stage 1: Extract tree data
        log_message("📤 Stage 1: Extracting tree data...", streamlit_container=streamlit_container)
        instance_name = get_instance_name(log_file)
        tree_sections = extract_tree_sections(log_file, debug=debug)
        
        if not tree_sections:
            raise ValueError(f"No tree data found in {log_file}")
        
        log_message(f"Found {len(tree_sections)} tree sections", 
                   streamlit_container=streamlit_container)
        
        # Create temporary tree detail file
        temp_dir = tempfile.mkdtemp(prefix="tree_analysis_")
        tree_filename = create_output_filename(instance_name)
        tree_detail_path = os.path.join(temp_dir, tree_filename)
        temp_files.append(tree_detail_path)
        
        tree_output = format_tree_report(tree_sections, instance_name)
        with open(tree_detail_path, 'w', encoding='utf-8') as f:
            f.write(tree_output)
        
        # Stage 2: Build structured tree
        log_message("🏗️  Stage 2: Building tree structure...", 
                   streamlit_container=streamlit_container)
        tree_nodes_raw = load_tree_data(tree_detail_path)
        
        if not tree_nodes_raw:
            raise ValueError(f"No nodes found in tree detail file")
        
        root = create_tree_hierarchy(tree_nodes_raw)
        if not root:
            raise ValueError(f"Could not build tree structure")
        
        level_sorted_nodes = sort_nodes_by_level(root)
        log_message(f"Built tree with {len(level_sorted_nodes)} nodes", 
                   streamlit_container=streamlit_container)
        
        # Create temporary structured tree file
        structured_filename = f"structured_tree_{instance_name}.txt"
        structured_tree_path = os.path.join(temp_dir, structured_filename)
        temp_files.append(structured_tree_path)
        
        structured_output = create_structured_report(level_sorted_nodes, instance_name)
        with open(structured_tree_path, 'w', encoding='utf-8') as f:
            f.write(structured_output)
        
        # Load the final tree structure
        tree_nodes, _ = load_structured_tree(structured_tree_path)
        
        if not tree_nodes:
            raise ValueError("Failed to load structured tree")
        
        log_message(f"Tree extraction completed: {len(tree_nodes)} nodes", 
                   "success", streamlit_container)
        
        # Create TreeData object
        if keep_intermediates:
            log_message(f"Intermediate files kept in: {temp_dir}", 
                       streamlit_container=streamlit_container)
            return TreeData(tree_nodes, instance_name, [])
        else:
            return TreeData(tree_nodes, instance_name, temp_files + [temp_dir])
            
    except Exception as e:
        # Clean up on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        log_message(f"Error during tree extraction: {e}", "error", streamlit_container)
        raise e


def plot_complex(tree, save_path=None, show_plot=True, figure_size=None, 
                streamlit_container=None):
    """
    Create complex tree visualization with estimation details and statistics.
    
    Args:
        tree (TreeData): Tree data from getTree()
        save_path (str): Path to save plot (optional)
        show_plot (bool): Whether to display plot interactively
        figure_size (tuple): Figure size (width, height)
        streamlit_container: Streamlit container for display (optional)
        
    Returns:
        dict: Plot statistics and information
        
    Example:
        # Console usage
        tree = getTree("solver.log")
        stats = plot_complex(tree, save_path="complex_tree.png")
        
        # Streamlit usage
        tree = getTree("solver.log", streamlit_container=st)
        stats = plot_complex(tree, streamlit_container=st)
    """
    if figure_size is None:
        figure_size = DEFAULT_FIGURE_SIZE
    
    log_message(f"🎨 Creating complex visualization for '{tree.instance_name}'",
               streamlit_container=streamlit_container)
    
    # Create graph and layout
    G = build_networkx_graph(tree.tree_nodes)
    pos = compute_layout_positions(G, tree.tree_nodes)
    
    # Get visualization elements
    colors, sizes = determine_node_appearance(tree.tree_nodes)
    labels = generate_node_labels(tree.tree_nodes)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', alpha=0.7, width=1.5,
                          connectionstyle="arc3,rad=0.1", ax=ax)
    
    # Draw nodes
    node_list = sorted(tree.tree_nodes.keys())
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.9, linewidths=2, edgecolors='black', ax=ax)
    
    # Draw optimal nodes
    optimal_nodes = [node_id for node_id, node_data in tree.tree_nodes.items() 
                    if getattr(node_data, 'is_optimal', False)]
    
    if optimal_nodes:
        optimal_positions = [pos[node_id] for node_id in optimal_nodes]
        optimal_x = [p[0] for p in optimal_positions]
        optimal_y = [p[1] for p in optimal_positions]
        
        ax.scatter(optimal_x, optimal_y, s=150, c='gold', marker='*', 
                  edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
    
    # Draw labels
    label_pos = {node: (x - 0.3, y) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels, font_size=9, font_weight='bold',
                          font_color='darkblue', verticalalignment='center',
                          horizontalalignment='right',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                          ax=ax)
    
    # Add level statistics
    add_level_stats(tree.tree_nodes, pos, ax)
    
    # Add estimation quality
    quality_stats = calculate_estimation_quality(tree.tree_nodes)
    all_stats = quality_stats['all']
    excl_stats = quality_stats['excluding_real_1']
    
    if all_stats['finite'] > 0:
        quality_text = "Estimation Quality:\n"
        quality_text += f"Good: {all_stats['good']} ({all_stats['good']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Over: {all_stats['over']} ({all_stats['over']/all_stats['finite']*100:.1f}%)\n"
        quality_text += f"Under: {all_stats['under']} ({all_stats['under']/all_stats['finite']*100:.1f}%)"
        
        if excl_stats['finite'] > 0:
            quality_text += f"\n\nExcl. real=1:\n"
            quality_text += f"Good: {excl_stats['good']} ({excl_stats['good']/excl_stats['finite']*100:.1f}%)"
        
        ax.text(0.98, 0.98, quality_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
                family='monospace')
    
    # Add legend
    legend_elements = create_legend_elements()
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # Set title
    ax.set_title(f"Complex B&B Tree Analysis: {tree.instance_name}\n"
                f"Node Format: (Real, Est), Size ∝ Real Count, Color = Quality\n"
                f"Level Stats: % Good Estimates (excluding trivial real=1 nodes)", 
                fontsize=12, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Handle output based on context
    if streamlit_container and STREAMLIT_AVAILABLE:
        # Streamlit display
        streamlit_container.pyplot(fig)
        
        # Optionally provide download button
        if save_path or True:  # Always offer download in Streamlit
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=DEFAULT_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            
            download_name = save_path or f"complex_tree_{tree.instance_name}.png"
            streamlit_container.download_button(
                label="📥 Download Complex Tree",
                data=buf.getvalue(),
                file_name=os.path.basename(download_name),
                mime="image/png"
            )
    else:
        # Traditional matplotlib handling
        if save_path:
            plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            log_message(f"Complex plot saved: {save_path}", "success")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'instance_name': tree.instance_name,
        'total_nodes': len(tree.tree_nodes),
        'optimal_nodes': len(optimal_nodes),
        'quality_stats': quality_stats
    }


def plot_simple(tree, save_path=None, show_plot=True, figure_size=None,
               streamlit_container=None):
    """
    Create simple tree visualization focusing on structure and termination reasons.
    
    Args:
        tree (TreeData): Tree data from getTree()
        save_path (str): Path to save plot (optional)
        show_plot (bool): Whether to display plot interactively
        figure_size (tuple): Figure size (width, height)
        streamlit_container: Streamlit container for display (optional)
        
    Returns:
        dict: Plot statistics and information
        
    Example:
        # Console usage
        tree = getTree("solver.log")
        stats = plot_simple(tree, save_path="simple_tree.png")
        
        # Streamlit usage
        tree = getTree("solver.log", streamlit_container=st)
        stats = plot_simple(tree, streamlit_container=st)
    """
    if figure_size is None:
        figure_size = SIMPLE_FIGURE_SIZE
    
    log_message(f"🎨 Creating simple visualization for '{tree.instance_name}'",
               streamlit_container=streamlit_container)
    
    # Create graph and layout
    G = build_networkx_graph(tree.tree_nodes)
    pos = compute_layout_positions(G, tree.tree_nodes)
    
    # Simple appearance
    colors, sizes = _get_simple_appearance(tree.tree_nodes)
    labels = _get_simple_labels(tree.tree_nodes)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=12, arrowstyle='->', alpha=0.6, width=1.0, ax=ax)
    
    # Draw nodes
    node_list = sorted(tree.tree_nodes.keys())
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                          node_color=colors, node_size=sizes, 
                          alpha=0.8, linewidths=1.5, edgecolors='black', ax=ax)
    
    # Draw optimal nodes
    optimal_nodes = [node_id for node_id, node_data in tree.tree_nodes.items() 
                    if getattr(node_data, 'is_optimal', False)]
    
    if optimal_nodes:
        optimal_positions = [pos[node_id] for node_id in optimal_nodes]
        optimal_x = [p[0] for p in optimal_positions]
        optimal_y = [p[1] for p in optimal_positions]
        
        ax.scatter(optimal_x, optimal_y, s=120, c='gold', marker='*', 
                  edgecolors='darkorange', linewidths=2, zorder=10, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                          font_color='darkblue', ax=ax)
    
    # Count termination reasons
    termination_counts = defaultdict(int)
    for node in tree.tree_nodes.values():
        termination_counts[getattr(node, 'termination_reason', 'unknown')] += 1
    
    # Add summary text
    summary_text = f"Nodes: {len(tree.tree_nodes)}\n"
    summary_text += f"Active: {termination_counts.get('active', 0)}\n"
    summary_text += f"Bound Pruned: {termination_counts.get('bound_pruned', 0)}\n"
    summary_text += f"Infeasible: {termination_counts.get('infeasible', 0)}\n"
    summary_text += f"Integer Solutions: {termination_counts.get('integer_solution', 0)}"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9),
            family='monospace')
    
    # Add legend
    legend_elements = _create_simple_legend()
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    
    # Set title
    ax.set_title(f"Simple B&B Tree Structure: {tree.instance_name}\n"
                f"Node Size ∝ Subtree Count, Color = Termination Reason", 
                fontsize=12, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Handle output based on context
    if streamlit_container and STREAMLIT_AVAILABLE:
        # Streamlit display
        streamlit_container.pyplot(fig)
        
        # Optionally provide download button
        if save_path or True:  # Always offer download in Streamlit
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=DEFAULT_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            
            download_name = save_path or f"simple_tree_{tree.instance_name}.png"
            streamlit_container.download_button(
                label="📥 Download Simple Tree",
                data=buf.getvalue(),
                file_name=os.path.basename(download_name),
                mime="image/png"
            )
    else:
        # Traditional matplotlib handling
        if save_path:
            plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            log_message(f"Simple plot saved: {save_path}", "success")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'instance_name': tree.instance_name,
        'total_nodes': len(tree.tree_nodes),
        'optimal_nodes': len(optimal_nodes),
        'termination_counts': dict(termination_counts)
    }


def _get_simple_appearance(tree_nodes):
    """Get colors and sizes for simple visualization."""
    color_map = {
        'active': 'lightblue',
        'bound_pruned': 'lightgreen',
        'infeasible': 'lightcoral',
        'integer_solution': 'gold',
        'unknown': 'lightgray'
    }
    
    actual_counts = [getattr(node, 'subtree_count', 1) for node in tree_nodes.values()]
    max_actual = max(actual_counts) if actual_counts else 1
    
    colors = []
    sizes = []
    
    for node_id in sorted(tree_nodes.keys()):
        node_data = tree_nodes[node_id]
        
        termination_reason = getattr(node_data, 'termination_reason', 'unknown')
        colors.append(color_map.get(termination_reason, 'lightgray'))
        
        subtree_count = getattr(node_data, 'subtree_count', 1)
        normalized_size = (subtree_count / max_actual) * NODE_SIZE_SCALE + DEFAULT_NODE_SIZE
        sizes.append(normalized_size)
    
    return colors, sizes


def _get_simple_labels(tree_nodes):
    """Get simple labels showing only node ID."""
    return {node_id: str(node_id) for node_id in tree_nodes.keys()}


def _create_simple_legend():
    """Create legend for simple visualization."""
    return [
        mpatches.Patch(color='lightblue', label='Active (Branched)'),
        mpatches.Patch(color='lightgreen', label='Bound Pruned'),
        mpatches.Patch(color='lightcoral', label='Infeasible'),
        mpatches.Patch(color='gold', label='Integer Solution'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=10, label='Optimal Node')
    ]


# Convenience functions
def analyze_tree_simple(log_file, save_path=None, show_plot=True, streamlit_container=None):
    """
    One-step function: extract tree and create simple visualization.
    
    Example:
        # Console
        analyze_tree_simple("solver.log", "simple_tree.png")
        
        # Streamlit
        analyze_tree_simple("solver.log", streamlit_container=st)
    """
    tree = getTree(log_file, streamlit_container=streamlit_container)
    return plot_simple(tree, save_path, show_plot, streamlit_container=streamlit_container)


def analyze_tree_complex(log_file, save_path=None, show_plot=True, streamlit_container=None):
    """
    One-step function: extract tree and create complex visualization.
    
    Example:
        # Console
        analyze_tree_complex("solver.log", "complex_tree.png")
        
        # Streamlit  
        analyze_tree_complex("solver.log", streamlit_container=st)
    """
    tree = getTree(log_file, streamlit_container=streamlit_container)
    return plot_complex(tree, save_path, show_plot, streamlit_container=streamlit_container)


def create_streamlit_app():
    """Example Streamlit app demonstrating usage."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: pip install streamlit")
        return
    
    st.title("🌳 Branch-and-Bound Tree Analysis")
    st.markdown("Upload a solver log file to analyze the B&B tree structure.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a solver log file", type=['txt', 'out', 'log'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        try:
            # Extract tree
            with st.spinner("Extracting tree structure..."):
                tree = getTree(temp_path, streamlit_container=st)
            
            # Visualization options
            st.subheader("Visualization Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Simple Tree View"):
                    st.subheader("Simple Tree Structure")
                    plot_simple(tree, streamlit_container=st)
            
            with col2:
                if st.button("🔬 Complex Tree Analysis"):
                    st.subheader("Complex Tree Analysis")
                    plot_complex(tree, streamlit_container=st)
            
            # Display tree information
            st.subheader("Tree Information")
            st.write(f"**Instance:** {tree.instance_name}")
            st.write(f"**Total Nodes:** {len(tree.tree_nodes)}")
            
            # Show termination reason breakdown
            termination_counts = defaultdict(int)
            for node in tree.tree_nodes.values():
                termination_counts[getattr(node, 'termination_reason', 'unknown')] += 1
            
            st.write("**Node Termination Breakdown:**")
            for reason, count in termination_counts.items():
                percentage = (count / len(tree.tree_nodes)) * 100
                st.write(f"- {reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                
        finally:
            # Clean up temporary file
            os.unlink(temp_path)


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Tree Analysis with Streamlit Support')
    parser.add_argument('log_file', nargs='?', help='Solver log file')
    parser.add_argument('--mode', choices=['simple', 'complex', 'both', 'streamlit'], default='simple',
                       help='Visualization mode')
    parser.add_argument('--save', help='Save path prefix')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    if args.mode == 'streamlit':
        print("🚀 Starting Streamlit app...")
        print("Run this command instead:")
        print("  streamlit run tree_analysis_streamlit.py")
        return
    
    if not args.log_file:
        print("🌳 Unified Tree Analysis with Streamlit Support")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python3 tree_analysis_streamlit.py <log_file> [options]")
        print("  streamlit run tree_analysis_streamlit.py")
        print()
        print("Examples:")
        print("  # Simple visualization")
        print("  python3 tree_analysis_streamlit.py solver.log --mode simple")
        print()
        print("  # Complex visualization")
        print("  python3 tree_analysis_streamlit.py solver.log --mode complex")
        print()
        print("  # Streamlit web app")
        print("  streamlit run tree_analysis_streamlit.py")
        return
    
    print("🌳 Unified Tree Analysis")
    print("=" * 50)
    
    # Extract tree
    tree = getTree(args.log_file)
    
    # Create visualizations
    if args.mode in ['simple', 'both']:
        save_path = f"{args.save}_simple.png" if args.save else None
        plot_simple(tree, save_path, not args.no_show)
    
    if args.mode in ['complex', 'both']:
        save_path = f"{args.save}_complex.png" if args.save else None
        plot_complex(tree, save_path, not args.no_show)
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - check if running in Streamlit
        try:
            # Check if we're in a Streamlit context
            import streamlit as st
            if hasattr(st, 'session_state'):
                # We're running in Streamlit - create the app
                create_streamlit_app()
            else:
                # Not in Streamlit - show help
                main()
        except:
            # Streamlit not available or other error - show help
            main()
    else:
        # Command line arguments provided
        main()