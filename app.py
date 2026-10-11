import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import build_model, compare_k, find_similar, load_data

st.set_page_config(page_title="News Topic Discovery", layout="wide")


@st.cache_resource(show_spinner="Training the model ...")
def get_model(k):
    return build_model(load_data(), k)          # runs once per k, then cached


@st.cache_data(show_spinner="Comparing cluster counts ...")
def get_k_table():
    return compare_k(get_model(5)["X"])


# ------------------------------------------------------------- sidebar
st.sidebar.title("News Intelligence")
page = st.sidebar.radio("Go to", ["Overview", "Topic Explorer", "Topic Visualization",
                                  "Similar News Search", "Model Evaluation"])
k = st.sidebar.slider("Number of topics", 2, 10, 5)

try:
    model = get_model(k)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

df = model["df"]
custom = st.session_state.setdefault("custom_names", {})
names = {c: f"Topic {c}: " + (custom.get((k, c)) or label) for c, label in model["labels"].items()}

# ------------------------------------------------------------- 1. Overview
if page == "Overview":
    st.title("News Intelligence & Topic Discovery")
    a, b, c, d = st.columns(4)
    a.metric("Articles processed", f"{len(df):,}")
    b.metric("Topics discovered", k)
    c.metric("Duplicates removed", load_data().attrs["n_raw"] - len(df))
    d.metric("Silhouette score", f"{model['silhouette']:.3f}")
    st.subheader("Workflow")
    st.code("News CSV -> clean text -> TF-IDF -> K-Means -> topic keywords\n"
            "                           |-> PCA map\n"
            "                           |-> cosine similarity search")

# --------------------------------------------------------- 2. Topic Explorer
elif page == "Topic Explorer":
    st.header("Topic Explorer")
    sizes = df["cluster"].value_counts()
    st.dataframe(pd.DataFrame({
        "Topic": [names[c] for c in sorted(names)],
        "Articles": [int(sizes.get(c, 0)) for c in sorted(names)],
        "Keywords": [", ".join(model["keywords"][c][:6]) for c in sorted(names)],
    }), hide_index=True, width="stretch")

    topic = st.selectbox("Choose a topic", sorted(names), format_func=lambda c: names[c])
    new_name = st.text_input("Rename this topic", value=custom.get((k, topic), model["labels"][topic]))
    custom[(k, topic)] = new_name.strip()

    st.markdown("**Keywords:** " + ", ".join(model["keywords"][topic]))
    closest = model["km"].transform(model["X"])[:, topic].argsort()[:3]   # nearest to the cluster centre
    st.markdown("**Representative headlines**")
    for i in closest:
        st.markdown(f"- {df.loc[i, 'title']}")
    st.dataframe(df[df["cluster"] == topic][["title"]], hide_index=True, width="stretch", height=300)

# ---------------------------------------------------- 3. Topic Visualization
elif page == "Topic Visualization":
    st.header("Topic Visualization")
    chosen = st.multiselect("Show topics", sorted(names), default=sorted(names),
                            format_func=lambda c: names[c])
    plot_df = df[df["cluster"].isin(chosen)].assign(Topic=lambda d: d["cluster"].map(names))
    fig = px.scatter(plot_df, x="x", y="y", color="Topic", hover_name="title",
                     hover_data={"x": False, "y": False, "Topic": False},
                     category_orders={"Topic": [names[c] for c in sorted(names)]},
                     labels={"x": "PCA component 1", "y": "PCA component 2"}, height=650)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"PCA squeezes the TF-IDF vectors into 2D ({model['variance'].sum():.1%} of the variance), "
               "so this map is only an approximation. Clustering used the full vectors.")

# ------------------------------------------------------ 4. Similar News Search
elif page == "Similar News Search":
    st.header("Similar News Search")
    mode = st.radio("Search by", ["Pick an existing article", "Enter new text"], horizontal=True)
    results = None
    if mode == "Pick an existing article":
        index = st.selectbox("Article", range(len(df)), format_func=lambda i: df.loc[i, "title"])
        results = find_similar(model, index=index)
    else:
        title = st.text_input("Title (optional)")
        text = st.text_area("Article text", height=150)
        if st.button("Find similar articles", type="primary"):
            try:
                results = find_similar(model, text=title + " " + text)
            except ValueError as error:
                st.warning(str(error))
    if results is not None:
        st.subheader("Top 5 similar articles")
        for _, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['title']}**")
                st.caption(f"{names[row['cluster']]}  |  similarity {row['similarity']:.3f}")
                st.write(row["text"][:220] + "...")

# ---------------------------------------------------------- 5. Model Evaluation
else:
    st.header("Model Evaluation")
    cols = st.columns(3)
    cols[0].metric("Silhouette (cosine)", f"{model['silhouette']:.3f}")
    cols[1].metric("Adjusted Rand Index", "n/a" if model["ari"] is None else f"{model['ari']:.3f}")
    cols[2].metric("Normalized Mutual Info", "n/a" if model["nmi"] is None else f"{model['nmi']:.3f}")

    sizes = df["cluster"].map(names).value_counts().reset_index()
    sizes.columns = ["Topic", "Articles"]
    st.plotly_chart(px.bar(sizes, x="Topic", y="Articles", title="Cluster sizes"), width="stretch")

    st.subheader("Cluster counts compared")
    st.dataframe(get_k_table(), hide_index=True)

    with st.expander("Limitations"):
        st.markdown("* Clustering is unsupervised: there is no accuracy score. ARI/NMI use the "
                    "dataset's labels only to check the result afterwards.\n"
                    "* A low silhouette is normal for text, and the highest score is not "
                    "automatically the best k - read the keywords.\n"
                    "* TF-IDF ignores word meaning; K-Means needs k in advance.")