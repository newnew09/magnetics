import streamlit as st

st.title("Hello👋")
st.markdown(
    """ 

    **I will :rainbow[show] you how to use it**\n



    """
)

if st.button("Send snows!"):
    st.snow()

if st.button("Send balloons!"):
    st.balloons()