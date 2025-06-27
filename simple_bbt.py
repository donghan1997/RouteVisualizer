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
    print(f"Picture saved at: {pic_path}")
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, Image


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
    
    print(f"🌳 Analyzing tree from: {os.path.basename(log_file)}")
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"❌ Error: File not found: {log_file}")
        return None
    
    # Check if bbt_plotter.py exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bbt_script = os.path.join(script_dir, "bbt_plotter.py")
    
    if not os.path.exists(bbt_script):
        print(f"❌ Error: bbt_plotter.py not found in {script_dir}")
        print("Please make sure bbt_plotter.py is in the same folder as this script")
        return None
    
    # Setup output folder
    if save_pic:
        if output_folder is None:
            output_folder = "tree_analysis"
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Saving results to: {output_folder}")
    
    # Build the command
    cmd = ["python3", bbt_script, log_file]
    
    if save_pic:
        cmd.extend(["--save-intermediates", "--output-dir", output_folder])
    
    if not show_tree:
        cmd.append("--no-show")
    
    try:
        print("🔄 Processing tree... (this may take a moment)")
        
        # Run the analysis
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              cwd=script_dir)
        
        if result.returncode == 0:
            print("✅ Tree analysis completed!")
            
            # Extract instance name from output for finding the saved picture
            if save_pic:
                # Look for the saved PNG file
                png_files = list(Path(output_folder).glob("bb_tree_analysis_*.png"))
                
                if png_files:
                    pic_path = str(png_files[0])
                    print(f"🖼️  Picture saved: {pic_path}")
                    
                    # Show the saved picture in notebook if not showing interactive plot
                    if not show_tree:
                        print("📊 Displaying saved tree:")
                        display(Image(filename=pic_path))
                    
                    return pic_path
                else:
                    print("⚠️  Warning: Picture was supposed to be saved but not found")
                    return None
            
            return None
            
        else:
            print("❌ Error during analysis:")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            return None
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return None


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
        print(f"Tree picture saved at: {pic_path}")
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
        print(f"Picture available at: {pic_path}")
    """
    return analyze_tree(log_file, save_pic=True, output_folder=output_folder, show_tree=True)


# Example usage and help
if __name__ == "__main__":
    print("🌳 Simple Branch-and-Bound Tree Analysis")
    print("=" * 50)
    print()
    print("This module provides simple functions to analyze B&B trees:")
    print()
    print("📋 Available functions:")
    print("  • analyze_tree(log_file)           - Main function")
    print("  • show_tree(log_file)              - Just show tree")
    print("  • save_tree_pic(log_file)          - Save picture only")
    print("  • get_tree_pic(log_file)           - Show + save")
    print()
    print("📖 Example usage:")
    print('  analyze_tree("output_2bwoeb_CVRP_120_1.txt")')
    print('  pic = save_tree_pic("my_log.txt", "my_pictures/")')
    print()
    print("💡 Tip: Import this in Jupyter notebook:")
    print("  from simple_bbt import analyze_tree, show_tree, save_tree_pic")