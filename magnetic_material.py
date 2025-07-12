import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def exponential_model(B, Cm, x, y, f):
    """指数型损耗模型: P = Cm * f^x * B^y"""
    return Cm * (f**x) * (B**y)


def material_dmr51w():
    st.title(':rainbow[磁性材料 DMR51W 损耗特性分析]')

    # 加载数据
    df = pd.read_csv("E:/Python-work/playground/dmr51w/data/DMR51W.csv")

    # 主界面筛选条件
    st.subheader("筛选条件")

    # 三列布局：频率、温度、磁通密度
    col1, col2, col3 = st.columns(3)

    with col1:
        # 频率选择器
        all_frequencies = sorted(df['Frequency (Hz)'].unique())
        default_freqs = all_frequencies[:2]  # 默认选择前两个频率
        selected_frequencies = st.multiselect(
            "选择频率 (Hz)",
            all_frequencies,
            default=default_freqs,
            key="frequency_selector"
        )

    with col2:
        # 温度选择器
        all_temperatures = sorted(df['Temperature (C)'].unique())
        selected_temperatures = st.multiselect(
            "选择温度 (°C)",
            all_temperatures,
            default=all_temperatures,
            key="temperature_selector"
        )

    with col3:
        # 磁通密度范围选择器
        min_b, max_b = df['Flux Density B (T)'].min(), df['Flux Density B (T)'].max()
        flux_range = st.slider(
            "磁通密度范围 (T)",
            min_value=float(min_b),
            max_value=float(max_b),
            value=(float(min_b), float(max_b)),
            step=0.01
        )

    # 筛选数据
    filtered_df = df[
        df['Frequency (Hz)'].isin(selected_frequencies) &
        df['Temperature (C)'].isin(selected_temperatures) &
        (df['Flux Density B (T)'] >= flux_range[0]) &
        (df['Flux Density B (T)'] <= flux_range[1])
    ]

    if filtered_df.empty:
        st.warning("没有数据匹配当前筛选条件，请调整筛选选项。")
        return

    # 指数模型拟合
    fit_results = {}
    for temp in selected_temperatures:
        temp_data = filtered_df[filtered_df['Temperature (C)'] == temp]
        if len(temp_data) < 3:
            continue

        # 提取B和P数据，频率固定为1kHz（用于归一化）
        B_data = temp_data['Flux Density B (T)']
        P_data = temp_data['Power Loss PV (W/m³)']
        f_data = temp_data['Frequency (Hz)']

        # 初始参数猜测
        p0 = [100, 1, 2]  # Cm, x, y

        try:
            # 使用curve_fit进行非线性最小二乘拟合
            popt, _ = curve_fit(
                lambda B, Cm, x, y: Cm * (f_data ** x) * (B ** y),
                B_data,
                P_data,
                p0=p0,
                maxfev=10000
            )
            fit_results[temp] = popt  # Cm, x, y
        except Exception as e:
            st.warning(f"温度 {temp}°C 的数据拟合失败: {str(e)}")
            fit_results[temp] = p0  # 使用初始猜测值

    # 如果没有成功拟合任何温度数据
    if not fit_results:
        st.warning("无法对选定数据进行拟合，请调整筛选条件。")
        return

    # 模型参数调整
    st.subheader("指数模型参数调整")

    # 选择要调整的温度
    selected_temp = st.selectbox(
        "选择温度进行参数调整",
        list(fit_results.keys())
    )

    # 获取初始拟合值
    initial_Cm, initial_x, initial_y = fit_results[selected_temp]

    # 创建参数调整输入框
    col4, col5, col6 = st.columns(3)

    with col4:
        Cm = st.number_input(
            f"Cm (初始值: {initial_Cm:.4f})",
            min_value=0.0,
            value=initial_Cm,
            step=0.1,
            format="%.4f"
        )

    with col5:
        x = st.number_input(
            f"x (频率指数, 初始值: {initial_x:.4f})",
            min_value=0.0,
            value=initial_x,
            step=0.01,
            format="%.4f"
        )

    with col6:
        y = st.number_input(
            f"y (磁通密度指数, 初始值: {initial_y:.4f})",
            min_value=0.0,
            value=initial_y,
            step=0.01,
            format="%.4f"
        )


    # 绘制散点图和趋势线
    st.subheader("损耗与磁通密度的关系")

    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_frequencies)))
    color_map = {freq: color for freq, color in zip(selected_frequencies, colors)}

    # 创建标记样式映射
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    marker_map = {temp: markers[i % len(markers)] for i, temp in enumerate(selected_temperatures)}

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制散点
    for freq in selected_frequencies:
        for temp in selected_temperatures:
            subset = filtered_df[
                (filtered_df['Frequency (Hz)'] == freq) &
                (filtered_df['Temperature (C)'] == temp)
            ]

            if not subset.empty:
                scatter = ax.scatter(
                    subset['Flux Density B (T)'],
                    subset['Power Loss PV (W/m³)'],
                    s=50,
                    color=color_map[freq],
                    marker=marker_map[temp],
                    alpha=0.7,
                    label=f'{freq} Hz, {temp}°C'
                )

    # 绘制趋势线（仅显示当前选中温度的调整后趋势线）
    B_plot = np.linspace(flux_range[0], flux_range[1], 100)

    for freq in selected_frequencies:
        # 计算调整后的模型预测值
        P_pred = exponential_model(B_plot, Cm, x, y, freq)

        # 显示当前参数的公式
        formula_text = f"P = {Cm:.4f} × f^{x:.4f} × B^{y:.4f}"
        ax.plot(
            B_plot, P_pred, '--',
            color=color_map[freq],
            alpha=0.8,
            label=f'趋势线: {formula_text}' if freq == selected_frequencies[0] else ""
        )

    # 添加图例
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 设置标题和坐标轴标签
    ax.set_title('单位体积磁芯损耗与磁通密度的关系')
    ax.set_xlabel('磁通密度 B (T)')
    ax.set_ylabel('单位体积磁芯损耗 PV (W/m^3)')

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 显示图表
    st.pyplot(fig)

    # 显示当前模型参数和拟合结果
    st.subheader("模型参数与拟合结果")
    st.markdown(f"当前温度: **{selected_temp}°C**")
    st.markdown(f"指数模型公式: **P = Cm × f^x × B^y**")
    st.markdown(f"参数值: Cm = **{Cm:.4f}**, x = **{x:.4f}**, y = **{y:.4f}**")

    # 数据详情
    with st.expander("查看详细数据"):
        st.dataframe(filtered_df)

    # 显示筛选后的数据行数
    st.info(f"当前显示: {len(filtered_df)} 个数据点 | 原始数据: {len(df)} 个数据点")