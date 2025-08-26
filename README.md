# Amazon-Fashion-Hybrid-Recommender-System
# 👗 Amazon Fashion Hybrid Recommender System

A **Hybrid Recommender System** built with **Streamlit** using the Amazon Fashion dataset.  
It combines **Popularity-Based**, **Content-Based (TF-IDF)**, and **Collaborative Filtering** approaches to provide smarter product recommendations.

---

## 🔥 Features
- **Popularity Recommendations** → Shows trending products based on ratings & review counts.  
- **Content-Based Recommendations** → Uses product descriptions (TF-IDF & cosine similarity) to suggest similar items.  
- **Collaborative Filtering** → Suggests items based on user–item interactions.  
- **Interactive Streamlit UI** → Dropdowns, buttons, and tables for easy exploration.  

---

## 📂 Dataset
We use the **Amazon Fashion Reviews and Metadata** dataset (converted from JSON → CSV).  

- `products_fashion.csv` → Product metadata (title, brand, category, description, price).  
- `reviews_fashion.csv` → Review details (user_id, product_id, rating, text, verified_purchase).  

👉 Dataset Reference: [Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)  

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/amazon-fashion-recommender.git
cd amazon-fashion-recommender

## Install dependencies:
```bash
pip install -r requirements.txt

### Usage
```bash
Run the Streamlit app: streamlit run app.py

### Tech Stack Python Streamlit Pandas, Scikit-Learn TF-IDF & Cosine Similarity Collaborative Filtering (user-item interactions).
