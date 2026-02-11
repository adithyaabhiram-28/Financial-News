# -------------------------------------------------------
# 🟣 News Topic Discovery Dashboard
# Hierarchical Clustering – DARK ANALYTICS EDITION
# -------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------------------------------
# Page Config + Dark Theme Injection
# -------------------------------------------------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS: Dark Analytics Aesthetic ----
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&family=Barlow+Condensed:wght@300;500;700&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    background-color: #0d0d0d !important;
    color: #e0e0e0 !important;
}

/* Main app background */
.stApp {
    background-color: #0d0d0d !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1e1e1e !important;
}

[data-testid="stSidebar"] * {
    color: #cccccc !important;
}

/* Sidebar sliders */
[data-testid="stSlider"] .stSlider > div > div {
    background: #1e4080 !important;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] {
    background-color: #0d0d0d !important;
    border-bottom: 1px solid #1a1a2e !important;
}

/* Title styling */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    border-bottom: 2px solid #00e5ff !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.2rem !important;
}

/* Subheader */
h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
    margin-bottom: 0.5rem !important;
    margin-top: 1.2rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111111 !important;
    border: 1px solid #1e1e1e !important;
    border-left: 3px solid #00e5ff !important;
    border-radius: 2px !important;
    padding: 0.8rem 1rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #666666 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #f0c14b !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
}

/* Info/Success/Warning boxes */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    border-left: 3px solid #00e5ff !important;
    background: #0a1628 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

/* Selectbox and other inputs */
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #cccccc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Paragraph / body text */
p, .stMarkdown p {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.95rem !important;
    color: #aaaaaa !important;
    letter-spacing: 0.03em !important;
}

/* Chart containers */
[data-testid="stpyplot"] {
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    background: #111111 !important;
}

/* Success / warning / error */
.element-container .stSuccess {
    background: #0a2010 !important;
    border-left: 3px solid #00ff88 !important;
    color: #00ff88 !important;
}
.element-container .stWarning {
    background: #1a1200 !important;
    border-left: 3px solid #f0c14b !important;
    color: #f0c14b !important;
}
.element-container .stError {
    background: #200a0a !important;
    border-left: 3px solid #ff4444 !important;
    color: #ff4444 !important;
}

/* Sidebar label */
[data-testid="stSidebar"] .stMarkdown {
    color: #888888 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Top metric row label */
.metric-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555555;
    margin-bottom: 2px;
}

.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00e5ff;
    line-height: 1;
}

.metric-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #444444;
}

/* KPI box */
.kpi-box {
    background: #111111;
    border: 1px solid #1e1e1e;
    border-top: 2px solid #00e5ff;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
}

.kpi-box.pink {
    border-top-color: #ff2d78;
}

.kpi-box.green {
    border-top-color: #00ff88;
}

.kpi-box.yellow {
    border-top-color: #f0c14b;
}

/* Section separator */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #444444;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 4px;
    margin-bottom: 12px;
    margin-top: 20px;
}

/* Chart title override */
.chart-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666666;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Chart palette — matches reference image
# -------------------------------------------------------
DARK_BG   = "#0d0d0d"
PANEL_BG  = "#111111"
CYAN      = "#00e5ff"
PINK      = "#ff2d78"
GREEN     = "#00ff88"
YELLOW    = "#f0c14b"
GRID_CLR  = "#1e1e1e"
AXIS_CLR  = "#333333"
TEXT_CLR  = "#555555"

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(PANEL_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=7)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color("#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.4, linestyle='-')
        ax.set_axisbelow(True)
    return fig

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.markdown('<div class="section-label">Vectorization</div>', unsafe_allow_html=True)

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 1000)
use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)
ngram_option = st.sidebar.selectbox("N-gram Range", ["Unigrams", "Bigrams", "Unigrams + Bigrams"])

st.sidebar.markdown('<div class="section-label">Clustering</div>', unsafe_allow_html=True)

linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
distance_metric = st.sidebar.selectbox("Distance Metric", ["euclidean"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

st.sidebar.markdown('<div class="section-label">Dendrogram</div>', unsafe_allow_html=True)
dendro_size = st.sidebar.slider("Articles for Dendrogram", 20, 200, 100)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("""
<h1>📡 NEWS TOPIC DISCOVERY</h1>
<p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#444; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.5rem;">
Hierarchical Clustering &nbsp;·&nbsp; Agglomerative &nbsp;·&nbsp; TF-IDF Vectorization
</p>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def get_ngram_range(option):
    if option == "Unigrams":      return (1, 1)
    elif option == "Bigrams":     return (2, 2)
    else:                         return (1, 2)

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
# Load Dataset
# -------------------------------------------------------
try:
    df = pd.read_csv("all-data.csv", header=None, encoding="latin1")
    df.columns = ["sentiment", "text"]
    st.sidebar.success(f"✓ all-data.csv  ·  {len(df):,} rows")
except:
    st.error("⚠  all-data.csv not found in project folder.")
    st.stop()

texts = df["text"].astype(str).replace("nan", "").str.strip()
texts = texts[texts != ""]
df = df.loc[texts.index].reset_index(drop=True)

# -------------------------------------------------------
# Vectorization
# -------------------------------------------------------
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

# -------------------------------------------------------
# Clustering
# -------------------------------------------------------
model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=distance_metric)
labels = model.fit_predict(X.toarray())
df["Cluster"] = labels
score = silhouette_score(X, labels)

top_terms = extract_top_terms_per_cluster(X.toarray(), labels, vectorizer, top_n=10)
summary_data = []
for c, keywords in top_terms:
    count = int(np.sum(labels == c))
    sample_text = texts[labels == c].iloc[0][:120]
    summary_data.append([c, count, keywords, sample_text])

# -------------------------------------------------------
# TOP KPI ROW  (like "Sessions / Session Length / PVs" row)
# -------------------------------------------------------
cluster_colors = [CYAN, PINK, GREEN, YELLOW, "#aa88ff", "#ff8844", "#44ccff", "#ffaa44", "#88ff44", "#ff44aa"]

kpi_cols = st.columns(3 + n_clusters)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="metric-title">Total Articles</div>
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-sub">corpus size</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-box pink">
        <div class="metric-title">TF-IDF Features</div>
        <div class="metric-value" style="color:{PINK}">{X.shape[1]:,}</div>
        <div class="metric-sub">vocabulary tokens</div>
    </div>""", unsafe_allow_html=True)

sil_color = GREEN if score > 0.5 else (YELLOW if score > 0 else "#ff4444")
with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-box green">
        <div class="metric-title">Silhouette Score</div>
        <div class="metric-value" style="color:{sil_color}">{score:.3f}</div>
        <div class="metric-sub">cluster quality</div>
    </div>""", unsafe_allow_html=True)

for i, row in enumerate(summary_data):
    c, count = row[0], row[1]
    pct = count / len(df) * 100
    col_idx = 3 + i
    if col_idx < len(kpi_cols):
        cc = cluster_colors[i % len(cluster_colors)]
        with kpi_cols[col_idx]:
            st.markdown(f"""
            <div class="kpi-box" style="border-top-color:{cc}">
                <div class="metric-title">Cluster {c}</div>
                <div class="metric-value" style="color:{cc}">{count:,}</div>
                <div class="metric-sub">{pct:.1f}% of corpus</div>
            </div>""", unsafe_allow_html=True)

# -------------------------------------------------------
# MAIN CHART ROW  (Dendrogram left, PCA right)
# -------------------------------------------------------
st.markdown('<div class="section-label">Structural Analysis</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="chart-title">Dendrogram &nbsp;·&nbsp; Hierarchical Link Structure</div>', unsafe_allow_html=True)

    subset_size = min(dendro_size, X.shape[0])
    X_subset = X[:subset_size].toarray()
    Z = linkage(X_subset, method=linkage_method)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    dendrogram(
        Z,
        ax=ax1,
        color_threshold=0,
        above_threshold_color=CYAN,
        leaf_font_size=0,
        orientation='bottom'
    )
    # Style dendrogram lines in cyan
    for coll in ax1.collections:
        coll.set_color(CYAN)
        coll.set_linewidth(0.6)
    for line in ax1.lines:
        line.set_color(CYAN)
        line.set_linewidth(0.6)

    ax1.set_xlabel("Article Index", fontsize=7, color=TEXT_CLR, labelpad=4)
    ax1.set_ylabel("Distance", fontsize=7, color=TEXT_CLR, labelpad=4)
    ax1.set_title("")

    # Add median reference line
    heights = Z[:, 2]
    median_h = np.median(heights)
    ax1.axhline(median_h, color=PINK, linewidth=0.8, linestyle='--', alpha=0.7)
    ax1.annotate(
        f"Median Distance: {median_h:.3f}",
        xy=(0.01, 0.96), xycoords='axes fraction',
        fontsize=6.5, color=PINK,
        fontfamily='monospace',
        ha='left', va='top'
    )

    apply_dark_style(fig1, [ax1])
    fig1.tight_layout(pad=0.8)
    st.pyplot(fig1)

    st.markdown(
        '<p style="font-size:0.7rem;color:#444;letter-spacing:0.1em;font-family:\'JetBrains Mono\',monospace;">'
        'LARGE VERTICAL GAPS → NATURAL TOPIC SEPARATION</p>',
        unsafe_allow_html=True
    )

with col_right:
    st.markdown('<div class="chart-title">PCA Projection &nbsp;·&nbsp; 2D Cluster Map</div>', unsafe_allow_html=True)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    for i in range(n_clusters):
        mask = labels == i
        cc = cluster_colors[i % len(cluster_colors)]
        ax2.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=cc,
            s=6,
            alpha=0.55,
            label=f"C{i}",
            linewidths=0
        )

    # Centroids
    for i in range(n_clusters):
        mask = labels == i
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        cc = cluster_colors[i % len(cluster_colors)]
        ax2.scatter(cx, cy, c=cc, s=60, marker='+', linewidths=1.2, zorder=5)
        ax2.annotate(
            f"C{i}",
            (cx, cy), fontsize=7.5, color=cc,
            fontfamily='monospace',
            xytext=(4, 4), textcoords='offset points'
        )

    exp_var = pca.explained_variance_ratio_
    ax2.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% var)", fontsize=7, color=TEXT_CLR)
    ax2.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% var)", fontsize=7, color=TEXT_CLR)
    ax2.legend(
        fontsize=6.5, framealpha=0,
        labelcolor='white',
        loc='upper right'
    )

    apply_dark_style(fig2, [ax2])
    fig2.tight_layout(pad=0.8)
    st.pyplot(fig2)

# -------------------------------------------------------
# BOTTOM ROW  (Cluster Size bars, term heatmap-style)
# -------------------------------------------------------
st.markdown('<div class="section-label">Cluster Intelligence</div>', unsafe_allow_html=True)

col_b1, col_b2 = st.columns([1, 2])

with col_b1:
    st.markdown('<div class="chart-title">Cluster Distribution</div>', unsafe_allow_html=True)

    cluster_ids = [row[0] for row in summary_data]
    cluster_counts = [row[1] for row in summary_data]
    bar_colors = [cluster_colors[i % len(cluster_colors)] for i in range(len(cluster_ids))]

    fig3, ax3 = plt.subplots(figsize=(4, 3.5))

    bars = ax3.bar(
        [f"C{c}" for c in cluster_ids],
        cluster_counts,
        color=bar_colors,
        width=0.55,
        zorder=3
    )

    # Add top-of-bar labels
    for bar, count in zip(bars, cluster_counts):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cluster_counts) * 0.02,
            f"{count:,}",
            ha='center', va='bottom',
            fontsize=7, color='#aaaaaa',
            fontfamily='monospace'
        )

    # Overlay line (like bounce rate line in reference)
    total = sum(cluster_counts)
    cumulative_pct = [sum(cluster_counts[:i+1]) / total * max(cluster_counts) for i in range(len(cluster_counts))]
    x_positions = range(len(cluster_ids))
    ax3.plot(x_positions, cumulative_pct, color=PINK, linewidth=1.2, marker='o', markersize=3, zorder=4)

    ax3.set_ylabel("Article Count", fontsize=7, color=TEXT_CLR)
    apply_dark_style(fig3, [ax3])
    fig3.tight_layout(pad=0.8)
    st.pyplot(fig3)

with col_b2:
    st.markdown('<div class="chart-title">Cluster Summary &nbsp;·&nbsp; Top Keywords &amp; Sample</div>', unsafe_allow_html=True)

    summary_df = pd.DataFrame(
        summary_data,
        columns=["Cluster", "Articles", "Top Keywords", "Sample"]
    )

    # Style dataframe to match dark theme
    st.dataframe(
        summary_df,
        use_container_width=True,
        height=200,
        column_config={
            "Cluster":  st.column_config.NumberColumn("ID", width="small"),
            "Articles": st.column_config.NumberColumn("Articles", width="small"),
            "Top Keywords": st.column_config.TextColumn("Top Keywords", width="large"),
            "Sample":   st.column_config.TextColumn("Sample Text", width="large"),
        }
    )

# -------------------------------------------------------
# Editorial Insights — compact monospace cards
# -------------------------------------------------------
st.markdown('<div class="section-label">Editorial Insights</div>', unsafe_allow_html=True)

insight_cols = st.columns(min(n_clusters, 5))
for i, row in enumerate(summary_data):
    cid, count, keywords, sample = row
    cc = cluster_colors[i % len(cluster_colors)]
    top_kws = [k.strip() for k in keywords.split(",")[:4]]
    col_idx = i % len(insight_cols)
    with insight_cols[col_idx]:
        tags = "  ".join([f"<span style='background:{cc}22;color:{cc};padding:1px 5px;font-size:0.65rem;letter-spacing:0.08em;border:1px solid {cc}44'>{k.upper()}</span>" for k in top_kws])
        st.markdown(f"""
        <div style="background:#111111;border:1px solid #1e1e1e;border-top:2px solid {cc};padding:0.8rem;margin-bottom:0.5rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;letter-spacing:0.2em;color:#555;text-transform:uppercase;margin-bottom:6px;">
                Cluster {cid} &nbsp;·&nbsp; {count:,} articles
            </div>
            <div style="margin-bottom:8px;line-height:1.8">{tags}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#444;line-height:1.4;border-top:1px solid #1e1e1e;padding-top:6px;margin-top:4px;">
                {sample[:90]}…
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("""
<div style="margin-top:2rem;border-top:1px solid #1e1e1e;padding-top:0.8rem;">
    <p style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#333;letter-spacing:0.1em;text-transform:uppercase;">
        Articles grouped by TF-IDF cosine similarity &nbsp;·&nbsp; Agglomerative ({linkage}) &nbsp;·&nbsp; PCA dimensionality reduction
    </p>
</div>
""".format(linkage=linkage_method), unsafe_allow_html=True)
