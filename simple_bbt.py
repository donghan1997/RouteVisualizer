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
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Try to import streamlit, fallback to print if not available
try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False

def log_info(message):
    """Helper function to log info - uses streamlit if available, otherwise print."""
    if USE_STREAMLIT:
        st.info(message)
    else:
        print(message)

def log_success(message):
    """Helper function to log success - uses streamlit if available, otherwise print."""
    if USE_STREAMLIT:
        st.success(message)
    else:
        print(f"✅ {message}")

def log_error(message):
    """Helper function to log error - uses streamlit if available, otherwise print."""
    if USE_STREAMLIT:
        st.error(message)
    else:
        print(f"❌ {message}")

def log_warning(message):
    """Helper function to log warning - uses streamlit if available, otherwise print."""
    if USE_STREAMLIT:
        st.warning(message)
    else:
        print(f"⚠️ {message}")


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
    
    log_info(f"🌳 Analyzing tree from: {os.path.basename(log_file)}")
    
    # Check if log file exists
    if not os.path.exists(log_file):
        log_error(f"File not found: {log_file}")
        return None
    
    # Check if tree_analysis_pipeline_simple.py exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bbt_script = os.path.join(script_dir, "tree_analysis_pipeline_simple.py")
    
    if not os.path.exists(bbt_script):
        log_error(f"tree_analysis_pipeline_simple.py not found in {script_dir}")
        log_error("Please make sure tree_analysis_pipeline_simple.py is in the same folder as this script")
        return None
    
    # Setup output folder
    if save_pic:
        if output_folder is None:
            output_folder = "tree_analysis"
        os.makedirs(output_folder, exist_ok=True)
        log_info(f"📁 Saving results to: {output_folder}")
    
    # Build the command
    cmd = [sys.executable, bbt_script, log_file]
    
    if save_pic:
        cmd.extend(["--save-intermediates", "--output-dir", output_folder])
    
    if not show_tree:
        cmd.append("--no-show")
    
    try:
        log_info("🔄 Processing tree... (this may take a moment)")
        
        # Clean up old PNG files
        if save_pic and output_folder:
            for f in Path(output_folder).glob("bb_tree_simple_*.png"):
                os.remove(f)
        
        result = subprocess.run(cmd, 
                                capture_output=True, 
                                text=True, 
                                cwd=script_dir,
                                timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            log_success("Tree analysis completed!")
            
            # Extract instance name from output for finding the saved picture
            if save_pic:
                # Look for the saved PNG file (corrected filename pattern)
                png_files = list(Path(output_folder).glob("bb_tree_simple_*.png"))
                
                if png_files:
                    pic_path = str(png_files[0])
                    log_success(f"Picture saved: {pic_path}")
                    
                    # Show the saved picture if streamlit is available and not showing interactive plot
                    if not show_tree and USE_STREAMLIT:
                        log_info("📊 Displaying saved tree:")
                        st.image(pic_path, caption="Branch-and-Bound Tree Analysis", use_column_width=True)
                    
                    return pic_path
                else:
                    log_warning("Picture was supposed to be saved but not found")
                    log_info("Looking for files in output directory:")
                    if output_folder and os.path.exists(output_folder):
                        files = list(os.listdir(output_folder))
                        log_info(f"Files found: {files}")
                    return None
            
            return None
            
        else:
            log_error("Error during analysis:")
            if result.stderr:
                log_error(f"Error output: {result.stderr}")
            if result.stdout:
                log_info(f"Standard output: {result.stdout}")
            return None
                
    except subprocess.TimeoutExpired:
        log_error("Analysis timed out (>5 minutes). The log file might be too large.")
        return None
    except Exception as e:
        log_error(f"Error running analysis: {e}")
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


def test_wrapper():
    """Test function to check if the wrapper works."""
    log_info("🧪 Testing simple_bbt wrapper...")
    
    # Check if required files exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "tree_analysis_pipeline_simple.py",
        "core/extract_tree_data.py",
        "core/build_tree_structure.py", 
        "core/visualize_tree.py",
        "visualize_tree_simple.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(script_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        log_error("Missing required files:")
        for file in missing_files:
            log_error(f"  - {file}")
        return False
    else:
        log_success("All required files found!")
        return True


# Example usage and help
if __name__ == "__main__":
    log_info("🌳 Simple Branch-and-Bound Tree Analysis")
    log_info("=" * 50)
    log_info("")
    log_info("This module provides simple functions to analyze B&B trees:")
    log_info("")
    log_info("📋 Available functions:")
    log_info("  • analyze_tree(log_file)           - Main function")
    log_info("  • show_tree(log_file)              - Just show tree")
    log_info("  • save_tree_pic(log_file)          - Save picture only")
    log_info("  • get_tree_pic(log_file)           - Show + save")
    log_info("  • test_wrapper()                   - Test if setup is correct")
    log_info("")
    log_info("📖 Example usage:")
    log_info('  analyze_tree("output_2bwoeb_CVRP_120_1.txt")')
    log_info('  pic = save_tree_pic("my_log.txt", "my_pictures/")')
    log_info("")
    log_info("💡 Tip: Import this in Jupyter notebook:")
    log_info("  from simple_bbt import analyze_tree, show_tree, save_tree_pic")
    log_info("")
    
    # Run test
    test_wrapper()