import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def flux_waveforms():
    st.title(':rainbow[磁通波形分析]')

    # 获取页面宽度的三分之一（假设页面宽度为 900 像素，可根据实际情况调整）
    page_width = 900
    img_width = page_width // 3

    # 创建三列布局，将第一张图片放在中间列以实现居中显示
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(r"E:\Python-work\playground\streamlit-demo\UIM.jpg", width=img_width)

    # 创建三列布局显示下方的三张图片
    col4, col5, col6 = st.columns(3)

    with col4:
        st.image(r"E:\Python-work\playground\streamlit-demo\SINE.jpg", width=img_width)
    with col5:
        st.image(r"E:\Python-work\playground\streamlit-demo\TWO_SQ.jpg", width=img_width)
    with col6:
        st.image(r"E:\Python-work\playground\streamlit-demo\THREE_SQ.jpg", width=img_width)

    # 用户选择激励电压波形
    waveform_choice = st.selectbox("选择激励电压波形", ["正弦波", "两电平方波", "三电平方波"])

    if waveform_choice == "正弦波":
        # 创建三列布局，将选中的正弦波图片放在中间列以实现居中显示
        col7, col8, col9 = st.columns([1, 1, 1])
        with col8:
            sine_img = st.image(r"E:\Python-work\playground\streamlit-demo\SINE.jpg", width=img_width, caption="选中的波形")
        # 加上单位，设置电感值精度到小数点后6位
        L = st.number_input("电感值 (μH)", value=1.0, step=0.000001, format="%.6f") * 1e-6  # 转换为 H
        f = st.number_input("频率 (kHz)", value=50.0) * 1e3  # 转换为 Hz
        U1 = st.number_input("电压U1 (V)", value=10.0)
        I_bias = st.number_input("电流偏置 (A)", value=0.0)

        # 生成 -0.5 到 1.5 个周期的时间序列
        t = np.linspace(-0.5/f, 1.5/f, 2000)
        # 计算电压和电流
        v = U1 * np.sin(2 * np.pi * f * t)
        i = (U1 / (2 * np.pi * f * L)) * (1 - np.cos(2 * np.pi * f * t))
        # 调整电流使得平均值为直流偏置值
        i = i - np.mean(i) + I_bias

    elif waveform_choice == "两电平方波":
        # 创建三列布局，将选中的两电平方波图片放在中间列以实现居中显示
        col10, col11, col12 = st.columns([1, 1, 1])
        with col11:
            two_sq_img = st.image(r"E:\Python-work\playground\streamlit-demo\TWO_SQ.jpg", width=img_width, caption="选中的波形")
        # 加上单位，设置电感值精度到小数点后6位
        L = st.number_input("电感值 (μH)", value=1.0, step=0.000001, format="%.6f") * 1e-6  # 转换为 H
        f = st.number_input("频率 (kHz)", value=50.0) * 1e3  # 转换为 Hz
        U1 = st.number_input("电压U1 (V)", value=10.0)
        D1 = st.number_input("占空比D1 (0 - 1之间)", min_value=0.0, max_value=1.0, value=0.5)
        I_bias = st.number_input("电流偏置 (A)", value=0.0)

        # 计算U2
        U2 = -U1 * D1 / (1 - D1)

        # 生成 -0.5 到 1.5 个周期的时间序列
        t = np.linspace(-0.5/f, 1.5/f, 2000)
        T = 1/f
        v = np.zeros_like(t)
        i = np.zeros_like(t)
        i[0] = 0

        for j in range(1, len(t)):
            if (t[j] % T) < D1 * T:
                v[j] = U1
            else:
                v[j] = U2

            dt = t[j] - t[j - 1]
            i[j] = i[j - 1] + (v[j] / L) * dt

        # 调整电流使得平均值为直流偏置值
        i = i - np.mean(i) + I_bias

    elif waveform_choice == "三电平方波":
        # 创建三列布局，将选中的三电平方波图片放在中间列以实现居中显示
        col13, col14, col15 = st.columns([1, 1, 1])
        with col14:
            three_sq_img = st.image(r"E:\Python-work\playground\streamlit-demo\THREE_SQ.jpg", width=img_width, caption="选中的波形")
        # 加上单位，设置电感值精度到小数点后6位
        L = st.number_input("电感值 (μH)", value=1.0, step=0.000001, format="%.6f") * 1e-6  # 转换为 H
        f = st.number_input("频率 (kHz)", value=50.0) * 1e3  # 转换为 Hz
        U1 = st.number_input("电压U1 (V)", value=10.0)
        U2 = st.number_input("电压U2 (V)", value=5.0)
        D1 = st.number_input("占空比D1 (0 - 1之间)", min_value=0.0, max_value=1.0, value=0.2)
        D2 = st.number_input("占空比D2 (大于0，小于1 - D1)", min_value=0.0, max_value=1.0 - D1, value=0.2)
        D3 = (1 - D1 - D2) / 2
        I_bias = st.number_input("电流偏置 (A)", value=0.0)

        # 计算U3
        U3 = -(U1 * D1 + U2 * D2) / (2 * D3)

        # 生成 -0.5 到 1.5 个周期的时间序列
        t = np.linspace(-0.5/f, 1.5/f, 2000)
        T = 1/f
        v = np.zeros_like(t)
        i = np.zeros_like(t)
        i[0] = 0

        for j in range(1, len(t)):
            t_mod = t[j] % T
            if t_mod < D1 * T:
                v[j] = U1
            elif t_mod < (D1 + D3) * T:
                v[j] = U3
            elif t_mod < (D1 + D3 + D2) * T:
                v[j] = U2
            else:
                v[j] = U3

            dt = t[j] - t[j - 1]
            i[j] = i[j - 1] + (v[j] / L) * dt

        # 调整电流使得平均值为直流偏置值
        i = i - np.mean(i) + I_bias

    # 设置图片风格，莫兰迪色系和字体调整
    plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # 绘制电压图，只显示第一个上升沿
    ax1.plot(t, v, color='#29A2C7', label='Voltage')  # 新的莫兰迪蓝色
    ax1.set_ylabel('Voltage (V)', color='#29A2C7')
    ax1.set_ylim(min(v) - 0.1 * abs(min(v)), max(v) + 0.1 * abs(max(v)))

    # 绘制电流图
    ax2.plot(t, i, color='#CA5D2A', label='Current')  # 新的莫兰迪红色
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (A)', color='#CA5D2A')
    ax2.set_ylim(min(i) - 0.1 * abs(min(i)), max(i) + 0.1 * abs(max(i)))

    # 找到电流的局部峰值和谷值点
    peaks, _ = find_peaks(i)
    valleys, _ = find_peaks(-i)
    critical_points = np.sort(np.concatenate((peaks, valleys)))

    # 添加对齐线到电流的局部峰值和谷值点
    for point in critical_points:
        x = t[point]
        ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

    # 给 0 - 1 个周期的背景添加浅黄色
    ax1.axvspan(0, 1/f, facecolor='yellow', alpha=0.3)
    ax2.axvspan(0, 1/f, facecolor='yellow', alpha=0.3)

    plt.tight_layout()  # 调整布局
    st.pyplot(fig)
