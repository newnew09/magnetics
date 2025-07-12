import streamlit as st
import os
from current_waveforms import waveforms_dcx
from magnetic_material import material_dmr51w
from flux_waveforms import flux_waveforms

def main():
    # 侧边栏功能选择
    st.sidebar.title('功能选择')
    page = st.sidebar.radio(' ', ['磁性材料分析','磁通波形分析','电流波形分析' ])

    if page == '电流波形分析':
      st.title(':rainbow[LLC谐振变换器简易波形分析]')

        # 插入图片
      image_path = r"E:\Python-work\playground\streamlit-demo\LLC.jpg"
      if os.path.exists(image_path):
            st.image(image_path, caption='LLC 谐振变换器 ', use_container_width=True)
      else:
            st.warning(f"未找到图片文件，请检查路径是否正确：\n {image_path}")

        # 创建参数输入区域
      st.header('参数设置')
      v_out = st.number_input('输出电压Vo (V)', value=12.0, step=1.0)
      i_out = st.number_input('输出电流Io (A, 平均值)', value=30.0, step=0.1)
      lm = st.number_input('励磁电感Lm (μH)', value=20.0, step=1.0, format="%.6f")
      freq = st.number_input('频率freq (kHz)', value=200.000, step=0.001)
      n = st.number_input('变比n (原边/副边)', value=4.0, step=0.1)

        # 计算按钮
      if st.button('计算波形'):
            waveforms_dcx(v_out, i_out, lm, freq, n)

    elif page == '磁性材料分析':
        material_dmr51w()

    elif page == '磁通波形分析':
        flux_waveforms()
    
    # 在侧边栏底部添加作者介绍
    st.sidebar.markdown("---")
    st.sidebar.write("Liu Yue")
    st.sidebar.write("liuyue0909@nuaa.edu.cn")
    st.sidebar.write("yueliu@cityu.edu.cn")
    st.sidebar.write("Unit: NUAA & CityU")

if __name__ == "__main__":
    main()