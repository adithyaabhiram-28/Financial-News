# app.py
# 🟣 News Topic Discovery Dashboard
# Using Hierarchical Clustering

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

st.title("🟣 News Topic Discovery Dashboard")
st.markdown(
    "This system groups similar news articles automatically based on textual similarity."
)

# -------------------------------------------------------
# Sidebar – Inputs
# -------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset (Optional)",
    type=["csv"]
)

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox(
    "Use English Stopwords",
    value=True
)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"]
)

dendro_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def get_ngram_range(option):
    if option == "Unigrams":
        return (1, 1)
    elif option == "Bigrams":
        return (2, 2)
    else:
        return (1, 2)


def extract_top_terms_per_cluster(X, labels, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    df_terms = []

    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        cluster_mean = X[idx].mean(axis=0)
        top_indices = np.argsort(cluster_mean)[::-1][:top_n]
        keywords = ", ".join([terms[i] for i in top_indices])

        df_terms.append((c, keywords))

    return df_terms


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        # Default dataset
        df = pd.read_csv("all-data.csv", header=None, encoding="latin1")
        df.columns = ["sentiment", "text"]
        st.sidebar.success("Loaded default dataset: all-data.csv")
    except:
        st.warning("Upload a dataset or place all-data.csv beside app.py")

if df is not None:

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # Detect text column automatically
    text_col = None
    for col in df.columns:
        if df[col].dtype == "object":
            text_col = col
            break

    texts = df[text_col].astype(str)

    # -------------------------------------------------------
    # Vectorization
    # -------------------------------------------------------
    ngram_range = get_ngram_range(ngram_option)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if use_stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(texts)

    st.write("TF-IDF Shape:", X.shape)

    # -------------------------------------------------------
    # Generate Dendrogram Button
    # -------------------------------------------------------
    if "Z" not in st.session_state:
        st.session_state.Z = None

    if st.button("🟦 Generate Dendrogram"):

        subset_size = min(dendro_size, X.shape[0])
        X_subset = X[:subset_size].toarray()

        Z = linkage(X_subset, method=linkage_method)
        st.session_state.Z = Z

    # -------------------------------------------------------
    # Dendrogram Section
    # -------------------------------------------------------
    if st.session_state.Z is not None:
        st.subheader("🌳 Dendrogram")

        fig = plt.figure(figsize=(12, 5))
        dendrogram(st.session_state.Z)
        plt.title("Dendrogram of News Articles")
        plt.xlabel("Article Index")
        plt.ylabel("Distance")
        st.pyplot(fig)

        st.info(
            "Large vertical gaps indicate natural separation between groups."
        )

    # -------------------------------------------------------
    # Apply Clustering
    # -------------------------------------------------------
    st.subheader("🟩 Apply Clustering")

    n_clusters = st.number_input(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    if st.button("Apply Clustering"):

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=distance_metric
        )

        labels = model.fit_predict(X.toarray())
        df["Cluster"] = labels

        # -------------------------------------------------------
        # PCA Visualization
        # -------------------------------------------------------
        st.subheader("📊 Cluster Visualization (PCA Projection)")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        fig2 = plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
        plt.title("Cluster Scatter Plot")
        st.pyplot(fig2)

        # -------------------------------------------------------
        # Cluster Summary
        # -------------------------------------------------------
        st.subheader("📋 Cluster Summary")

        top_terms = extract_top_terms_per_cluster(
            X.toarray(),
            labels,
            vectorizer,
            top_n=10
        )

        summary_data = []

        for c, keywords in top_terms:
            count = np.sum(labels == c)
            sample_text = texts[labels == c].iloc[0][:120]
            summary_data.append([c, count, keywords, sample_text])

        summary_df = pd.DataFrame(
            summary_data,
            columns=[
                "Cluster ID",
                "Number of Articles",
                "Top Keywords",
                "Sample Article"
            ]
        )

        st.dataframe(summary_df)

        # -------------------------------------------------------
        # Silhouette Score
        # -------------------------------------------------------
        st.subheader("📊 Validation")

        score = silhouette_score(X, labels)
        st.metric("Silhouette Score", round(score, 4))

        if score > 0.5:
            st.success("Clusters are clearly separated.")
        elif score > 0:
            st.warning("Some overlap exists between clusters.")
        else:
            st.error("Clusters may not be meaningful.")

        # -------------------------------------------------------
        # Editorial Insights
        # -------------------------------------------------------
        st.subheader("🧠 Editorial Insights")

        for row in summary_data:
            cid = row[0]
            keywords = row[2].split(",")[:3]
            st.write(
                f"🟣 Cluster {cid}: Articles appear to focus on topics related to "
                f"{', '.join(keywords)}."
            )

        # -------------------------------------------------------
        # Guidance Box
        # -------------------------------------------------------
        st.info(
            "Articles grouped together share similar vocabulary and themes. "
            "These groupings can support tagging, recommendations, and content organization."
        )
