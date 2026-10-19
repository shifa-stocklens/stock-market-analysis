import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="StockLens",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #10283a);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #091522;
}
h1, h2, h3, p, label {
    color: #eef5ff;
}
div.stButton > button {
    background-color: #35c6a4;
    color: #07111f;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    st.title("📈 StockLens")
    st.header("Stock Market Analysis")
    st.subheader("Welcome Back")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username.strip().lower() == "shifa" and password == "12345":
            st.session_state["login"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.caption("Demo username: shifa")
    st.caption("Demo password: 12345")

    st.stop()
