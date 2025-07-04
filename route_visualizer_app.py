import streamlit as st
import matplotlib.pyplot as plt
import json
from tree_analysis_unified import getTree, plot_simple, plot_complex
import os
import contextlib
import io

st.set_page_config(
    page_title="CombiView",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)


st.title("CombiView")
st.caption("A Visual Toolkit for Combinatorial Optimization Problems")

# problem_type = st.sidebar.selectbox(
#     "Select Problem Type",
#     ["VRPs", "Crew Scheduling(Coming Soon)", "Facility Location(Coming Soon)"]
# )
problem_options_raw = [
    "VRPs",
    "Crew Scheduling (Coming Soon)",
    "Facility Location (Coming Soon)"
]

# Sidebar radio selector
selected = st.sidebar.radio(
    "Select Problem Type",
    options=problem_options_raw,
)

if selected == "VRPs":
    st.header("Route Visualizer for CVRP")

    files = st.file_uploader("Upload your route & tree files", type=["json", "txt"], accept_multiple_files=True)
    uploaded_file = next((f for f in files if f.name.endswith('.json')), None)
    bbt_file = next((f for f in files if f.name.endswith('.txt')), None)


    if uploaded_file is not None:
        data = json.load(uploaded_file)
        data_available = True
    else:
        # st.info("No JSON uploaded, using example data.")
        with open("example/example.json") as f:
            data = json.load(f)
        data_available = True
        # st.info("Using built-in example data.")

    if bbt_file is not None:
        temp_path = os.path.join("temp", bbt_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(bbt_file.getvalue())
        bbt_available = True
    else:
        default_bbt_path = "example/example_bbt.txt"
        if os.path.exists(default_bbt_path):
            temp_path = default_bbt_path
            bbt_available = True
            # st.info("Using built-in example branch-and-bound file.")



            # 解析数据字段
    coordinates = data.get("nodes", [])
    standard_routes = data.get("standard", [])
    minmax_routes = data.get("minmax", [])
    range_routes = data.get("range", [])

    xy_coords = [(pt['x'], pt['y']) for pt in coordinates]
    x, y = zip(*xy_coords)

    tab1, tab2, tab3, tab4 = st.tabs([
        "0. Integrated View",
        "1. Standard CVRP Routes",
        "2. Branch-and-Bound Tree",
        "3. Fairness-based Routes"
    ])


    with tab1:
        st.subheader("Integrated View")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("### Standard Routes")

            if data_available:
            # 示例图
                fig, ax = plt.subplots(figsize=(6, 6))

                # 绘制所有节点
                
                ax.scatter(x, y, color='black', s=15)
                # for idx, (x_val, y_val) in enumerate(xy_coords):
                #     ax.text(x_val + 0.5, y_val + 0.5, str(idx), fontsize=4)

                for i, route in enumerate(standard_routes['routes']):
                    path = route['path']
                    route_coords = [xy_coords[0]]  # Start from depot
                    route_coords += [xy_coords[p] for p in path]
                    route_coords.append(xy_coords[0])  # Return to depot
                    rx, ry = zip(*route_coords)
                    ax.plot(rx[1:-1], ry[1:-1], marker='o', linestyle='-',zorder=1)
                    ax.plot([rx[0], rx[1]], [ry[0], ry[1]], color="gray", linestyle='--',zorder=0)
                    ax.plot([rx[-2], rx[-1]], [ry[-2], ry[-1]], color="gray", linestyle='--',zorder=0)

                ax.set_title("Standard CVRP Routes")
                ax.axis('equal')
                st.pyplot(fig)

        with col2:
            st.markdown("### Branch-and-Bound Tree")

            if bbt_available:

                # st.info("Analyzing branch-and-bound tree...")
                with contextlib.redirect_stdout(io.StringIO()):
                    tree = getTree(temp_path, debug=False, keep_intermediates=False, streamlit_container=None)
                    simple_plot_path = os.path.join("temp/", "simple_tree.png")
                    stats = plot_simple(tree, save_path=simple_plot_path, show_plot=False, figure_size=(5,5))


                st.image(simple_plot_path, use_container_width=True)
            


        with col3:
            st.markdown("### Fairness-based Routes")

            if data_available:
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.scatter(x, y, color='black', s=15)

                selected_routes = range_routes["routes"]
        
                for i, route in enumerate(selected_routes):
                    path = route["path"]
                    path = route['path']
                    route_coords = [xy_coords[0]]  # Start from depot
                    route_coords += [xy_coords[p] for p in path]
                    route_coords.append(xy_coords[0])  # Return to depot
                    rx, ry = zip(*route_coords)
                    # use different line styles for depot to first point and last point to depot
                    ax2.plot(rx[1:-1], ry[1:-1], marker='o', linestyle='-',zorder=1)
                    ax2.plot([rx[0], rx[1]], [ry[0], ry[1]], color="gray", linestyle='--',zorder=0)
                    ax2.plot([rx[-2], rx[-1]], [ry[-2], ry[-1]], color="gray", linestyle='--',zorder=0)
                ax2.set_title("Range Fairness Routes")
                ax2.set_aspect("equal")
                st.pyplot(fig2)



    # 1. 标准 CVRP 路径图
    with tab2:
        
        st.subheader("Standard CVRP Path")
        st.write("Visualize the optimal routes for the standard CVRP problem.")
        
        if data_available:
            # 示例图
            fig, ax = plt.subplots(figsize=(8, 8))

            # 绘制所有节点
            
            ax.scatter(x, y, color='black', s=15)
            # for idx, (x_val, y_val) in enumerate(xy_coords):
            #     ax.text(x_val + 0.5, y_val + 0.5, str(idx), fontsize=4)

            for i, route in enumerate(standard_routes['routes']):
                path = route['path']
                route_coords = [xy_coords[0]]  # Start from depot
                route_coords += [xy_coords[p] for p in path]
                route_coords.append(xy_coords[0])  # Return to depot
                rx, ry = zip(*route_coords)
                ax.plot(rx[1:-1], ry[1:-1], marker='o', linestyle='-',zorder=1)
                ax.plot([rx[0], rx[1]], [ry[0], ry[1]], color="gray", linestyle='--',zorder=0)
                ax.plot([rx[-2], rx[-1]], [ry[-2], ry[-1]], color="gray", linestyle='--',zorder=0)

            ax.set_title("Standard CVRP Routes")
            ax.axis('equal')
            st.pyplot(fig)

    # 2. 分支限界树图
    with tab3:
        st.subheader("Branch-and-Bound Tree")
        st.write("Visualize the decision process during CVRP solving.")

        st.info("This section displays the branch-and-bound tree structure.")
        # 示例图像
        # 上传 B&B 输出文件

        if bbt_available:
            # 保存上传文件到临时路径
            # temp_path = os.path.join("temp", bbt_file.name)
            # os.makedirs("temp", exist_ok=True)
            # with open(temp_path, "wb") as f:
            #     f.write(bbt_file.getvalue())

            # 分析并保存图像
            st.info("Analyzing branch-and-bound tree...")

            tree = getTree(temp_path, debug=False, keep_intermediates=False, streamlit_container=st)
            simple_plot_path = os.path.join("temp/", "simple_tree.png")
            plot_simple(tree, save_path=simple_plot_path, show_plot=False)
            st.image(simple_plot_path, caption="🧩 Simple B&B Tree", use_container_width=True)

            if st.checkbox("Detailed Tree Visualization", value=False):

                complex_plot_path = os.path.join("temp/", "complex_tree.png")
                plot_complex(tree, save_path=complex_plot_path, show_plot=False)
                st.image(complex_plot_path, caption="🔍 Detailed B&B Tree", use_container_width=True)

            # st.image(simple_plot_path, caption="Branch-and-Bound Tree", use_container_width=True)

            # st.image(complex_plot_path, caption="Branch-and-Bound Tree", use_container_width=True)

        else:
            st.info("Please upload a B&B output file (e.g., .txt format).")

    with tab4:
        st.subheader("Routes under Fairness Principles vs. Standard CVRP")

        # 用户选择公平性策略
        fairness_option = st.selectbox(
            "Select fairness principle", 
            ["Min-max", "Range"]
        )

        st.write(f"Comparing **Standard CVRP Route** with **{fairness_option} Fairness Route**")

        # 示例：定义标准路径


        if data_available:
            col1, col2 = st.columns([1, 1])


            
            # ------- 左图：标准路径 -------
            with col1:
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.scatter(x, y, color='black', s=15)
                for i, route in enumerate(standard_routes['routes']):
                    path = route['path']
                    route_coords = [xy_coords[0]]  # Start from depot
                    route_coords += [xy_coords[p] for p in path]
                    route_coords.append(xy_coords[0])  # Return to depot
                    rx, ry = zip(*route_coords)
                    # use different line styles for depot to first point and last point to depot
                    ax1.plot(rx[1:-1], ry[1:-1], marker='o', linestyle='-',zorder=1)
                    ax1.plot([rx[0], rx[1]], [ry[0], ry[1]], color="gray", linestyle='--',zorder=0)
                    ax1.plot([rx[-2], rx[-1]], [ry[-2], ry[-1]], color="gray", linestyle='--',zorder=0)

                ax1.set_title("Standard CVRP Routes")
                ax1.axis('equal')
                st.pyplot(fig1)

            # ------- 右图：公平路径 -------
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.scatter(x, y, color='black', s=15)

                selected_routes = (
                    minmax_routes["routes"] if fairness_option == "Min-max" else range_routes["routes"]
                )
                for i, route in enumerate(selected_routes):
                    path = route["path"]
                    path = route['path']
                    route_coords = [xy_coords[0]]  # Start from depot
                    route_coords += [xy_coords[p] for p in path]
                    route_coords.append(xy_coords[0])  # Return to depot
                    rx, ry = zip(*route_coords)
                    # use different line styles for depot to first point and last point to depot
                    ax2.plot(rx[1:-1], ry[1:-1], marker='o', linestyle='-',zorder=1)
                    ax2.plot([rx[0], rx[1]], [ry[0], ry[1]], color="gray", linestyle='--',zorder=0)
                    ax2.plot([rx[-2], rx[-1]], [ry[-2], ry[-1]], color="gray", linestyle='--',zorder=0)
                ax2.set_title(f"{fairness_option} Fairness Routes")
                ax2.set_aspect("equal")
                st.pyplot(fig2)
        else:
            st.info("Please upload a JSON file first.")