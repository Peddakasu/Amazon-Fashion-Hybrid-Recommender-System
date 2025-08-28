import os
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ===============================
# 1. Cached Loader (local + URL only)
# ===============================
@st.cache_data
def load_dataset_safe(file_name, url=None):
    """
    Try loading dataset from local disk or URL.
    If fails, return None (handled outside).
    """
    # --- Load from local ---
    if os.path.exists(file_name):
        try:
            return pd.read_csv(file_name)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file_name} locally: {e}")

    # --- Download from URL ---
    if url:
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(file_name, "wb") as f:
                f.write(r.content)
            return pd.read_csv(file_name)
        except Exception as e:
            st.warning(f"⚠️ Download failed for {file_name}: {e}")

    return None  # fallback handled outside

# --- Products dataset ---
products = load_dataset_safe(
    "products_fashion_sample.csv",
    url="https://drive.google.com/uc?id=19MQoHjhYDd-X1X3f4ABn4b5HyG-Hy5RZ&export=download"
)
if products is None:
    st.info("📂 Upload products_fashion.csv")
    uploaded_products = st.file_uploader("Upload products dataset", type=["csv", "zip"])
    if uploaded_products:
        if uploaded_products.name.endswith(".zip"):
            products = pd.read_csv(uploaded_products, compression="zip")
        else:
            products = pd.read_csv(uploaded_products)
    else:
        st.stop()

# --- Reviews dataset ---
reviews = load_dataset_safe(
    "reviews_fashion_sample.csv",
    url="https://drive.google.com/uc?id=1LSIYlvaO-uEk3qPccCMwo_SE1WGNEBI-&export=download"
)
if reviews is None:
    st.info("📂 Upload reviews_fashion.csv")
    uploaded_reviews = st.file_uploader("Upload reviews dataset", type=["csv", "zip"])
    if uploaded_reviews:
        if uploaded_reviews.name.endswith(".zip"):
            reviews = pd.read_csv(uploaded_reviews, compression="zip")
        else:
            reviews = pd.read_csv(uploaded_reviews)
    else:
        st.stop()
#st.write("Products columns:", products.columns.tolist())
#st.write("Reviews columns:", reviews.columns.tolist())
#st.write("First 5 rows of products:", products.head())
#st.write("First 5 rows of reviews:", reviews.head())


# ===============================
# 2. Detect Text Columns (Multi-Column Handling)
# ===============================
def detect_text_columns(df):
    candidate_cols = ["description", "product_description", "title", "product_title", "brand"]
    found_cols = [col for col in candidate_cols if col in df.columns]
    if not found_cols:
        df["combined_text"] = "product"
        return "combined_text", []
    if len(found_cols) == 1:
        return found_cols[0], found_cols
    # Combine multiple text columns into one
    df["combined_text"] = df[found_cols].fillna("").agg(" ".join, axis=1)
    return "combined_text", found_cols

text_column, used_cols = detect_text_columns(products)

# ===============================
# 3. Build TF-IDF Matrix (Resilient)
# ===============================
@st.cache_resource
def build_tfidf_safe(df):
    # Validate text content
    df[text_column] = df[text_column].astype(str).fillna("")
    if df[text_column].str.strip().eq("").all():
        df[text_column] = "product"
        st.info("Using placeholder text for empty content.")
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df[text_column])
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("Empty vocabulary")
        return tfidf, tfidf_matrix
    except Exception as e:
        st.warning(f"TF-IDF vectorization failed: {e}")
        return None, None

tfidf, tfidf_matrix = build_tfidf_safe(products)

content_based_enabled = tfidf is not None and tfidf_matrix is not None

# ===============================
# 4. Ensure Product Titles
# ===============================
if "product_title" not in products.columns:
    products["product_title"] = [f"Product {i+1}" for i in range(len(products))]

# ===============================
# 5. Popularity Recommender
# ===============================
def get_popular_products(n=10):
   if "product_id" not in reviews.columns:
        st.warning("⚠️ 'product_id' not found in reviews. Popularity recommender disabled.")
        return pd.DataFrame()
    popular = (
        reviews.groupby("product_id")
        .agg(rating_mean=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    if "product_id" in products.columns:
        popular = popular.merge(products, on="product_id", how="left")
    popular = popular.sort_values(["rating_count", "rating_mean"], ascending=False)
    return popular.head(n)

# ===============================
# 6. Content-Based Recommender (Safe)
# ===============================
def content_recommender(selected_product, top_n=5):
    if not content_based_enabled:
        return pd.DataFrame()
    indices = pd.Series(products.index, index=products["product_title"]).drop_duplicates()
    if selected_product not in indices:
        return pd.DataFrame()
    idx = indices[selected_product]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return products.iloc[product_indices]

# ===============================
# 7. Collaborative Filtering (Safe)
# ===============================
def collaborative_recommender(user_id, top_n=5):
    if "user_id" not in reviews.columns or "product_id" not in reviews.columns:
        return pd.DataFrame()
    user_ratings = reviews[reviews["user_id"] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()
    top_product = user_ratings.sort_values("rating", ascending=False).iloc[0]["product_id"]
    similar_users = reviews[reviews["product_id"] == top_product]["user_id"].unique()
    collab = reviews[reviews["user_id"].isin(similar_users)]
    collab_avg = collab.groupby("product_id")["rating"].mean().reset_index()
    collab = collab_avg.merge(products, on="product_id", how="left")
    collab = collab.sort_values("rating", ascending=False)
    return collab.head(top_n)

# ===============================
# 8. Streamlit UI
# ===============================
st.title("👗 Amazon Fashion Hybrid Recommender System")

# Display popular products
popular = get_popular_products(10)
if not popular.empty:
    st.header("🔥 Popular Products")
    st.dataframe(popular[["product_id", "product_title", "category", "brand", "rating_mean", "rating_count"]])

# Product selection
st.header("🎯 Select a Product for Recommendations")
product_choice = st.selectbox("Choose a product:", products["product_title"].values)

# Show selected product details
if product_choice:
    selected = products[products["product_title"] == product_choice].iloc[0]
    st.markdown(
        f"**Product ID:** {selected['product_id']}  \n"
        f"**Title:** {selected['product_title']}  \n"
        f"**Category:** {selected.get('category', 'N/A')}  \n"
        f"**Brand:** {selected.get('brand', 'N/A')}"
        f"**Mean Rating:** {selected['rating_mean']:.2f} ⭐  \n"
        f"**Total Reviews:** {selected['rating_count']}"
    )

# Button to trigger recommendations
if st.button("🔍 Get Recommendations") and product_choice:
    # Content-Based
    st.subheader("📖 Content-Based Recommendations")
    content_recs = content_recommender(product_choice)
    if content_recs.empty:
        st.warning("Content-based recommendations not available.")
    else:
        st.dataframe(content_recs[["product_title", "category", "brand", "price"]])

    # Collaborative Filtering
    st.subheader("👥 Collaborative Filtering Recommendations")
    if "user_id" in reviews.columns:
        # Find an example user who rated the selected product highly
        sample_user_df = reviews[
            (reviews["product_id"] == selected.get("product_id"))
        ]
        if not sample_user_df.empty:
            user_id = sample_user_df.iloc[0]["user_id"]
            collab_recs = collaborative_recommender(user_id)
            if collab_recs.empty:
                st.warning("No collaborative recommendations found.")
            else:
                st.dataframe(collab_recs[["product_title", "rating"]])
        else:
            st.warning("No collaborative recommendations available.")
