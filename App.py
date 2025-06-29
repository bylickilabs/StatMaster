import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("StatMaster: Interaktive Data Science Analyse")

# Daten-Upload
uploaded_file = st.file_uploader("CSV/Excel Datei hochladen", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("**Datenvorschau:**")
    st.dataframe(df)

    # Statistische Kennzahlen
    st.header("Deskriptive Statistik")
    st.write(df.describe())
