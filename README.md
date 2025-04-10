
# DuckDB as a vector database

```bash
uv run python -m src.db_setup
```

Once the database has been initialised and populated, you can explore it using a simple duckdb web UI:

```bash
duckdb -ui local_files/vector_db.duckdb
```

Start the frontend:
```bash
uv run streamlit run streamlit_frontend.py
```

Using the search functions directly in python:
```python
import json
from src import search
from typing import Final

USER_QUERY: Final[str] = "yesterday apples"

bm25_df = search.bm25(
  user_query=USER_QUERY,
  top_k=5,
  output_format="polars",
)
bm25_df.to_dicts()

semantic_search_results = search.semantic(
  user_query=USER_QUERY,
  top_k=5,
  output_format="python_list",
)
print(
  json.dumps(
    semantic_search_results,
    indent=4,
    default=str,
  )
)

hybrid_search_df = search.hybrid_rrf(
  user_query=USER_QUERY,
  prefetch_k=100,
  top_k=5,
  output_format="polars",
)
hybrid_search_df.to_dicts()
```

