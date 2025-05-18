"""
The main entry script of the streamlit web app frontend
"""

import streamlit as st

from src import constants, search

st.set_page_config(page_title="duckdb-as-vecdb", layout="wide")
st.title("DuckDB as a Vector Database")

search_method = st.radio("Select Search Method", ["BM25", "semantic", "hybrid (RRF)"])

user_query: str = st.text_input("Enter your search query")

top_k = st.number_input(
    "Number of results to return", min_value=1, max_value=999, value=5
)

if st.button("Search") and user_query:
    if search_method == "BM25":
        results_df = search.bm25(
            user_query=user_query,
            top_k=top_k,
            output_format="polars",
        )
    elif search_method == "semantic":
        results_df = search.semantic(
            user_query=user_query, top_k=top_k, output_format="polars"
        )
    elif search_method == "hybrid (RRF)":
        results_df = search.hybrid_rrf(
            user_query=user_query,
            prefetch_k=500,
            top_k=top_k,
            output_format="polars",
        )

    st.subheader("Search Results")
    st.dataframe(results_df)

st.markdown("---")
st.caption("Select a method, enter a query, and hit Search to view results.")
