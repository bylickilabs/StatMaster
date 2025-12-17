import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="StatMaster",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="darkgrid")

st.title("📊 StatMaster")
st.caption("Professionelle Data-Science Analyseplattform")

@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    try:
        return pd.read_csv(
            file,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file,
            sep=None,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip"
        )

uploaded_file = st.file_uploader(
    "CSV- oder Excel-Datei hochladen",
    type=["csv", "xlsx"]
)

if not uploaded_file:
    st.info("Bitte laden Sie eine Datei hoch, um die Analyse zu starten.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error("Die Datei konnte nicht verarbeitet werden.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("Die Datei enthält keine verwertbaren Daten.")
    st.stop()

num_cols = df.select_dtypes(include=np.number).columns.tolist()

if not num_cols:
    st.warning("Keine numerischen Spalten für Analysen verfügbar.")
    st.stop()

st.header("🔍 Datenvorschau")
st.dataframe(df, use_container_width=True)

st.header("📈 Deskriptive Statistik")
st.dataframe(df[num_cols].describe())

st.header("🔗 Korrelationsmatrix")

if len(num_cols) <= 15:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Zu viele numerische Spalten für eine übersichtliche Darstellung.")

st.header("📊 Visualisierung")

c1, c2 = st.columns(2)

with c1:
    x_col = st.selectbox("X-Achse", num_cols)
    y_col = st.selectbox("Y-Achse", num_cols)

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax1)
    st.pyplot(fig1)
    plt.close(fig1)

with c2:
    hist_col = st.selectbox("Histogramm", num_cols)

    fig2, ax2 = plt.subplots()
    sns.histplot(df[hist_col], kde=True, ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)

st.header("🧠 K-Means Clustering")

if st.checkbox("Clustering aktivieren"):
    cluster_cols = st.multiselect(
        "Spalten für Clustering",
        num_cols
    )

    if len(cluster_cols) < 2:
        st.warning("Mindestens zwei numerische Spalten auswählen.")
    else:
        k = st.slider("Cluster-Anzahl", 2, 10, 3)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cluster_cols])

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        df["cluster"] = model.fit_predict(scaled)

        st.success("Clustering erfolgreich durchgeführt.")

        st.dataframe(df[["cluster"] + cluster_cols])

        fig3, ax3 = plt.subplots()
        sns.scatterplot(
            x=df[cluster_cols[0]],
            y=df[cluster_cols[1]],
            hue=df["cluster"],
            palette="deep",
            ax=ax3
        )
        st.pyplot(fig3)
        plt.close(fig3)

st.markdown("---")
st.caption("StatMaster · Data Science · Analytics · Streamlit")
