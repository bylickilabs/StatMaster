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

   # Korrelationsmatrix
    st.header("Korrelationsmatrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

   # Visualisierung
    st.header("Visualisierung")
    col1, col2 = st.columns(2)
    with col1:
        col_x = st.selectbox("X-Achse", df.columns)
        col_y = st.selectbox("Y-Achse", df.columns)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax2)
        st.pyplot(fig2)
    with col2:
        col_hist = st.selectbox("Spalte für Histogramm", df.columns)
        fig3, ax3 = plt.subplots()
        sns.histplot(df[col_hist], kde=True, ax=ax3)
        st.pyplot(fig3)

   # Einfaches Clustering (optional)
    if st.checkbox("K-Means Clustering durchführen"):
        from sklearn.cluster import KMeans
        n_clusters = st.slider("Cluster-Anzahl", 2, 8, 3)
        cols = st.multiselect("Spalten für Clustering", df.select_dtypes(include=np.number).columns.tolist())
        if len(cols) >= 2:
            km = KMeans(n_clusters=n_clusters)
            df['cluster'] = km.fit_predict(df[cols])
            st.write(df[['cluster'] + cols])
            fig4, ax4 = plt.subplots()
            sns.scatterplot(x=df[cols[0]], y=df[cols[1]], hue=df['cluster'], palette="deep", ax=ax4)
            st.pyplot(fig4)

else:
    st.info("Bitte laden Sie eine CSV- oder Excel-Datei hoch, um zu starten.")
