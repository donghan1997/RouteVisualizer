#!/usr/bin/env python3
"""
Simple Branch-and-Bound Tree Analysis Function

A user-friendly wrapper for BBT analysis that hides all the complexity.
Just call the function with a log file and get the tree visualization!

Usage:
    from simple_bbt import analyze_tree
    
    # Basic usage - just show the tree
    analyze_tree("output_2bwoeb_CVRP_120_1.txt")
    
    # Save the picture too
    pic_path = analyze_tree("output_2bwoeb_CVRP_120_1.txt", save_pic=True)
    st.info(f"Picture saved at: {pic_path}")
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
# from IPython.display import display, Image
import streamlit as st


def analyze_tree(log_file, save_pic=False, output_folder=None, show_tree=True):
    """
    Analyze a branch-and-bound tree from solver log and show/save the visualization.
    
    Args:
        log_file (str): Path to the solver log file (e.g., "output_2bwoeb_CVRP_120_1.txt")
        save_pic (bool): Whether to save the tree picture (default: False)
        output_folder (str): Where to save files (default: creates "tree_analysis/" folder)
        show_tree (bool): Whether to display the tree (default: True)
    
    Returns:
        str or None: Path to saved picture if save_pic=True, otherwise None
    
    Examples:
        # Just show the tree
        analyze_tree("my_solver_output.txt")
        
        # Show tree and save picture
        pic_path = analyze_tree("my_solver_output.txt", save_pic=True)
        
        # Save to specific folder
        analyze_tree("my_solver_output.txt", save_pic=True, output_folder="my_results/")
    """
    
    st.info(f"🌳 Analyzing tree from: {os.path.basename(log_file)}")
    
    # Check if log file exists
    if not os.path.exists(log_file):
        st.info(f"❌ Error: File not found: {log_file}")
        st.info(f"❌ Error: File not found: {log_file}")
        return None
    
    # Check if bbt_plotter.py exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bbt_script = os.path.join(script_dir, "bbt_plotter.py")
    
    if not os.path.exists(bbt_script):
        st.info(f"❌ Error: bbt_plotter.py not found in {script_dir}")
        st.info("Please make sure bbt_plotter.py is in the same folder as this script")
        return None
    
    # Setup output folder
    if save_pic:
        if output_folder is None:
            output_folder = "tree_analysis"
        os.makedirs(output_folder, exist_ok=True)
        st.info(f"📁 Saving results to: {output_folder}")
    
    # Build the command
    cmd = ["/home/adminuser/venv/bin/python3", bbt_script, log_file]
    
    if save_pic:
        cmd.extend(["--save-intermediates", "--output-dir", output_folder])
    
    if not show_tree:
        cmd.append("--no-show")
    
    # try:
    st.info("🔄 Processing tree... (this may take a moment)")
    
    # Run the analysis
    result = subprocess.run(cmd, 
                            capture_output=True, 
                            text=True, 
                            cwd=script_dir)
    
    if result.returncode == 0:
        st.info("✅ Tree analysis completed!")
        
        # Extract instance name from output for finding the saved picture
        if save_pic:
            # Look for the saved PNG file
            png_files = list(Path(output_folder).glob("bb_tree_analysis_*.png"))
            
            if png_files:
                pic_path = str(png_files[0])
                st.info(f"🖼️  Picture saved: {pic_path}")
                
                # Show the saved picture in notebook if not showing interactive plot
                if not show_tree:
                    st.info("📊 Displaying saved tree:")
                    # display(Image(filename=pic_path))
                    return pic_path 
                    # st.image(pic_path, caption="Branch-and-Bound Tree Analysis", use_column_width=True)
                
                return pic_path
            else:
                st.info("⚠️  Warning: Picture was supposed to be saved but not found")
                return None
        
        return None
        
    else:
        st.info("❌ Error during analysis:")
        if result.stderr:
            st.info(result.stderr)
        if result.stdout:
            st.info(result.stdout)
        return None
            
    # except Exception as e:
    #     st.info(f"❌ Error running analysis: {e}")
    #     return None


def show_tree(log_file):
    """
    Quick function to just show the tree (no saving).
    
    Args:
        log_file (str): Path to the solver log file
    
    Example:
        show_tree("output_2bwoeb_CVRP_120_1.txt")
    """
    return analyze_tree(log_file, save_pic=False, show_tree=True)


def save_tree_pic(log_file, output_folder="tree_pictures"):
    """
    Function to analyze tree and save picture without showing it.
    
    Args:
        log_file (str): Path to the solver log file
        output_folder (str): Where to save the picture
    
    Returns:
        str: Path to the saved picture
    
    Example:
        pic_path = save_tree_pic("output_2bwoeb_CVRP_120_1.txt")
        st.info(f"Tree picture saved at: {pic_path}")
    """
    return analyze_tree(log_file, save_pic=True, output_folder=output_folder, show_tree=False)


def get_tree_pic(log_file, output_folder="tree_analysis"):
    """
    Function to analyze tree, save picture, and return the path.
    Shows the tree AND saves it.
    
    Args:
        log_file (str): Path to the solver log file
        output_folder (str): Where to save the picture
    
    Returns:
        str: Path to the saved picture
    
    Example:
        pic_path = get_tree_pic("output_2bwoeb_CVRP_120_1.txt")
        st.info(f"Picture available at: {pic_path}")
    """
    return analyze_tree(log_file, save_pic=True, output_folder=output_folder, show_tree=True)


# Example usage and help
if __name__ == "__main__":
    st.info("🌳 Simple Branch-and-Bound Tree Analysis")
    st.info("=" * 50)
    st.info()
    st.info("This module provides simple functions to analyze B&B trees:")
    st.info()
    st.info("📋 Available functions:")
    st.info("  • analyze_tree(log_file)           - Main function")
    st.info("  • show_tree(log_file)              - Just show tree")
    st.info("  • save_tree_pic(log_file)          - Save picture only")
    st.info("  • get_tree_pic(log_file)           - Show + save")
    st.info()
    st.info("📖 Example usage:")
    st.info('  analyze_tree("output_2bwoeb_CVRP_120_1.txt")')
    st.info('  pic = save_tree_pic("my_log.txt", "my_pictures/")')
    st.info()
    st.info("💡 Tip: Import this in Jupyter notebook:")
    st.info("  from simple_bbt import analyze_tree, show_tree, save_tree_pic")