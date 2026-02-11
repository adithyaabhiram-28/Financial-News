# -------------------------------------------------------
# 🟣 News Topic Discovery Dashboard
# Hierarchical Clustering – AUTO VERSION
# -------------------------------------------------------

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
st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.markdown(
    "This system groups similar news articles automatically based on textual similarity."
)

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
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

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 3
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
    output = []

    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        cluster_mean = X[idx].mean(axis=0)
        top_indices = np.argsort(cluster_mean)[::-1][:top_n]
        keywords = ", ".join([terms[i] for i in top_indices])

        output.append((c, keywords))

    return output


# -------------------------------------------------------
# Load Dataset (AUTO)
# -------------------------------------------------------
try:
    df = pd.read_csv("all-data.csv", header=None, encoding="latin1")
    df.columns = ["sentiment", "text"]
    st.sidebar.success("Loaded dataset: all-data.csv")
except:
    st.error("all-data.csv not found in project folder.")
    st.stop()

# Clean text safely
texts = df["text"].astype(str).replace("nan", "").str.strip()
texts = texts[texts != ""]
df = df.loc[texts.index].reset_index(drop=True)

# -------------------------------------------------------
# Vectorization
# -------------------------------------------------------
st.subheader("TF-IDF Vectorization")

ngram_range = get_ngram_range(ngram_option)

try:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if use_stopwords else None,
        ngram_range=ngram_range,
        min_df=2
    )

    X = vectorizer.fit_transform(texts)

    if X.shape[1] == 0:
        st.error("Vocabulary is empty. Disable stopwords or change n-grams.")
        st.stop()

except:
    st.error("TF-IDF failed. Try different settings.")
    st.stop()

st.write("TF-IDF Shape:", X.shape)

# -------------------------------------------------------
# 🌳 Dendrogram
# -------------------------------------------------------
st.subheader("Dendrogram")

subset_size = min(dendro_size, X.shape[0])
X_subset = X[:subset_size].toarray()

Z = linkage(X_subset, method=linkage_method)

fig = plt.figure(figsize=(12,5))
dendrogram(Z)
plt.xlabel("Article Index")
plt.ylabel("Distance")
st.pyplot(fig)

st.info(
    "Large vertical gaps suggest natural topic separation."
)

# -------------------------------------------------------
# 🟩 Agglomerative Clustering
# -------------------------------------------------------
st.subheader("Clustering Output")

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
st.subheader("PCA Cluster Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

fig2 = plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
plt.title("Cluster Projection")
st.pyplot(fig2)

# -------------------------------------------------------
# Cluster Summary Table
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
# Validation Section
# -------------------------------------------------------
st.subheader("📊 Validation")

score = silhouette_score(X, labels)
st.metric("Silhouette Score", round(score, 4))

if score > 0.5:
    st.success("Clusters are clearly separated.")
elif score > 0:
    st.warning("Clusters have some overlap.")
else:
    st.error("Clusters may not be meaningful.")

# -------------------------------------------------------
# Editorial Insights
# -------------------------------------------------------
st.subheader("Editorial Insights")

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
    "These clusters can support automatic tagging, recommendations, "
    "and content organization."
)
