
import streamlit as st

st.title("Hello QQ 👋")
st.markdown(
    """ 

    **I will :rainbow[love] you forever!**

    """
)

if st.button("Send balloons!"):
    st.balloons()
