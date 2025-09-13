import streamlit as st
import pandas as pd

@st.cache_data
def load_hotels(path: str):
    df = pd.read_csv(path)
    if "Hotel_ID" in df.columns:
        df["Hotel_ID"] = df["Hotel_ID"].astype(str)
    return df
