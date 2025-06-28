import streamlit as st
import matplotlib.pyplot as plt
import json
from simple_bbt import analyze_tree
import os

st.set_page_config(layout="wide", page_title="CVRP Visualization")

st.title("CVRP Visualization Dashboard")

uploaded_file = st.file_uploader("Upload CVRP Route JSON File", type="json")

if uploaded_file is not None:
    # 加载 JSON 数据
    data = json.load(uploaded_file)

    # 解析数据字段
    coordinates = data.get("nodes", [])
    standard_routes = data.get("standard", [])
    minmax_routes = data.get("minmax", [])
    range_routes = data.get("range", [])

    xy_coords = [(pt['x'], pt['y']) for pt in coordinates]
    x, y = zip(*xy_coords)

else:
    st.info("Please upload a JSON file to begin.")

tab1, tab2, tab3 = st.tabs(["1. Standard CVRP Path", "2. Branch-and-Bound Tree", "3. Fairness-based Routes"])


# 1. 标准 CVRP 路径图
with tab1:
    
    st.subheader("Standard CVRP Path")
    st.write("Visualize the optimal route for the standard CVRP problem.")
    
    if uploaded_file:
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
with tab2:
    st.subheader("Branch-and-Bound Tree")
    st.write("Visualize the decision process during CVRP solving.")

    st.info("This section displays the branch-and-bound tree structure.")
    # 示例图像
    # 上传 B&B 输出文件
    bbt_file = st.file_uploader("Upload Branch-and-Bound Output File", type="txt")

    if bbt_file is not None:
        # 保存上传文件到临时路径
        temp_path = os.path.join("temp", bbt_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(bbt_file.getvalue())

        # 分析并保存图像
        st.info("Analyzing branch-and-bound tree...")
        pic_path = analyze_tree(temp_path, save_pic=True)
        print(f"Tree image saved to: {pic_path}")
        # 显示图像
        # if os.path.exists(pic_path):
        #     st.image(pic_path, caption="Branch-and-Bound Tree", use_container_width=True)
        # else:
        #     st.error("Tree image not found.")
        try:
            st.image(pic_path, caption="Branch-and-Bound Tree", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to display image: {e}")
    else:
        st.info("Please upload a B&B output file (e.g., .txt format).")

with tab3:
    st.subheader("Routes under Fairness Principles vs. Standard CVRP")

    # 用户选择公平性策略
    fairness_option = st.selectbox(
        "Select fairness principle", 
        ["Min-max", "Range"]
    )

    st.write(f"Comparing **Standard CVRP Route** with **{fairness_option} Fairness Route**")

    # 示例：定义标准路径


    if uploaded_file:
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