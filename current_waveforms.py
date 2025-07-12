import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def waveforms_dcx(v_out, i_out, lm, freq, n):
        
    try:
        # 将电感单位从 μH 转换为 H，频率单位从 kHz 转换为 Hz
        lm = lm * 1e-6
        freq = freq * 1e3

        # 1. 计算励磁电流峰值
        im_peak = n * v_out / (4 * freq * lm)
        import streamlit as st
        st.subheader('计算参数检查（负载过小时结果不准确）')
        st.write(f'励磁电流峰值 = {im_peak:.2f}A')

        # 2. 设置时间参数
        fs = 5000 * freq
        T = 1 / freq
        t = np.arange(0, 2 * T + 1 / fs, 1 / fs)

        # 3. 定义计算平均电流的函数
        def calc_aligned_avg_current(a):
            # 生成波形
            i_primary = a * np.sin(2 * np.pi * freq * t)

            # 生成正确的正负对称三角波
            i_mag = im_peak * 2 * np.abs(2 * (freq * t - np.floor(freq * t + 0.5))) - im_peak

            # 找到正弦波下降沿中值等于im_peak的点
            # 首先找到正弦波的峰值点
            peak_indices = np.where(np.diff(np.sign(np.diff(i_primary))) == -2)[0] + 1
            if len(peak_indices) == 0:
                # 如果没有找到峰值，使用最大值索引
                peak_idx = np.argmax(i_primary)
            else:
                peak_idx = peak_indices[0]

            # 在峰值点之后寻找第一个小于等于im_peak的点
            idx_after_peak = np.where(i_primary[peak_idx:] <= im_peak)[0]
            if len(idx_after_peak) > 0:
                target_idx = peak_idx + idx_after_peak[0]
            else:
                # 如果没有找到，使用峰值点
                target_idx = peak_idx

            # 找到三角波的第一个峰值点
            mag_peak_indices = np.where(np.diff(np.sign(np.diff(i_mag))) == -2)[0] + 1
            if len(mag_peak_indices) > 0:
                first_mag_peak_idx = mag_peak_indices[0]
            else:
                first_mag_peak_idx = np.argmax(i_mag)

            # 对齐波形：使三角波的峰值与正弦波上值等于im_peak的点对齐
            shift = target_idx - first_mag_peak_idx
            i_mag_aligned = np.roll(i_mag, shift)

            # 计算副边电流和平均值
            i_secondary = n * (i_primary - i_mag_aligned)
            # 排除初始过渡，计算平均值
            avg_current = np.mean(np.abs(i_secondary[len(i_secondary) // 4:]))
            return avg_current

        # 4. 求解原边电流幅值
        search_range = [im_peak, 50 * im_peak]  # 扩大初始搜索范围

        # 检查函数值符号
        f_low = calc_aligned_avg_current(search_range[0]) - i_out
        f_high = calc_aligned_avg_current(search_range[1]) - i_out

        st.write(f'输出电流最小值为 [{calc_aligned_avg_current(search_range[0]):.2f}A] ')

        max_attempts = 5  # 最大尝试次数
        attempt = 0
        while np.sign(f_low) == np.sign(f_high) and attempt < max_attempts:
            search_range[1] = search_range[1] * 2
            f_high = calc_aligned_avg_current(search_range[1]) - i_out
            attempt += 1

        if np.sign(f_low) == np.sign(f_high):
            st.error('无解：请检查参数，建议调整:\n'
                     '1. 增大励磁电感 Lm\n'
                     '2. 降低输出电流要求 Iout\n'
                     '3. 调整变比 n')
            return

        # 使用fsolve求解方程
        def equation(a):
            return calc_aligned_avg_current(a) - i_out

        primary_amp = fsolve(equation, search_range[0], xtol=1e-6)[0]

        # 5. 生成最终对齐的波形
        i_primary = primary_amp * np.sin(2 * np.pi * freq * t)

        # 生成正确的正负对称三角波
        i_mag = im_peak * 2 * np.abs(2 * (freq * t - np.floor(freq * t + 0.5))) - im_peak

        # 找到正弦波下降沿中值等于im_peak的点
        peak_indices = np.where(np.diff(np.sign(np.diff(i_primary))) == -2)[0] + 1
        if len(peak_indices) == 0:
            peak_idx = np.argmax(i_primary)
        else:
            peak_idx = peak_indices[0]

        # 在峰值点之后寻找第一个小于等于im_peak的点
        idx_after_peak = np.where(i_primary[peak_idx:] <= im_peak)[0]
        if len(idx_after_peak) > 0:
            target_idx = peak_idx + idx_after_peak[0]
        else:
            target_idx = peak_idx

        # 找到三角波的第一个峰值点
        mag_peak_indices = np.where(np.diff(np.sign(np.diff(i_mag))) == -2)[0] + 1
        if len(mag_peak_indices) > 0:
            first_mag_peak_idx = mag_peak_indices[0]
        else:
            first_mag_peak_idx = np.argmax(i_mag)

        # 对齐波形
        shift = target_idx - first_mag_peak_idx
        i_mag = np.roll(i_mag, shift)

        # 6. 计算副边电流
        i_secondary = n * (i_primary - i_mag)
        avg_secondary = np.mean(np.abs(i_secondary[len(i_secondary) // 4:]))

        # 7. 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 绘制原边和励磁电流
        ax1.plot(t, i_primary, 'b-', linewidth=3, label='Primary current')
        ax1.plot(t, i_mag, 'r-', linewidth=3, label='Magnetizing current')
        ax1.plot(t[target_idx], i_primary[target_idx], 'ko', markerfacecolor='k')
        ax1.axhline(y=im_peak, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f'Primary current({primary_amp:.2f}A) vs Magnetizing current(Peak{im_peak:.2f}A)', fontsize=18)
        ax1.set_xlabel('Time (s)', fontsize=18)
        ax1.set_ylabel('Current (A)', fontsize=18)
        # 调整坐标轴刻度字体大小
        ax1.tick_params(axis='both', which='major', labelsize=16)  # 主刻度字体大小
        ax1.tick_params(axis='both', which='minor', labelsize=14)  # 副刻度字体大小（可选）
        # 调整图例字体大小
        ax1.legend(fontsize=16, loc='lower left')  # loc参数可选，控制图例位置
        ax1.grid(True)
        ax1.set_xlim([0, 2 * T])
        # ax1.legend()

        # 绘制副边电流
        ax2.plot(t, i_secondary, 'g-', linewidth=3, label='Secondary current')
        ax2.plot(t, np.abs(i_secondary), 'm--', linewidth=3, label='Absolute')
        ax2.plot([t[0], t[-1]], [avg_secondary, avg_secondary], 'k:', linewidth=3, label='Average')
        ax2.set_title(f'Secondary current (Average={avg_secondary:.3f}A)', fontsize=18)
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.set_ylabel('Current (A)', fontsize=18)
        # 调整坐标轴刻度字体大小
        ax2.tick_params(axis='both', which='major', labelsize=16)  # 主刻度字体大小
        ax2.tick_params(axis='both', which='minor', labelsize=14)  # 副刻度字体大小（可选）
        # 调整图例字体大小
        ax2.legend(fontsize=16, loc='lower left')  # loc参数可选，控制图例位置
        ax2.grid(True)
        ax2.set_xlim([0, 2 * T])
        # ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"计算过程中发生错误: {str(e)}")