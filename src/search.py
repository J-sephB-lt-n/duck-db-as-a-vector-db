from typing import Final, Literal

import duckdb
import polars as pl

from src import constants
from src.language_models import embed_model

def bm25(
    user_query: str,
    top_k: int = 10,
    output_format: Literal["python_list", "polars"] = "python_list",
) -> list[dict] | pl.DataFrame:
    """
    Return `top_k` closest results to `user_query` using BM25 (full-text search)
    """
    SUPPORTED_OUTPUT_FORMATS: Final[list[str]] = ["python_list", "polars"]
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format='{output_format}' is not currently supported" +
            "\nAvailable output formats are [" +
            ",".join([f"'{x}'" for x in SUPPORTED_OUTPUT_FORMATS]) +
            "]"
        )
    sql_query: str = """
        SELECT      'fts_bm25' AS 'search_method'
                ,   row_id 
                ,   msg_text
                ,   score
                ,   ROW_NUMBER() OVER (ORDER BY score DESC, RANDOM()) AS rank
        FROM    (
                    SELECT  *
                        ,   fts_main_main.MATCH_BM25(
                                row_id,
                                $user_query,
                                fields := 'msg_text'
                            ) AS score
                    FROM    main
                ) bm 
        WHERE       score IS NOT NULL
        ORDER BY    score DESC
        LIMIT       $top_k 
        ;
    """
    with duckdb.connect(constants.DB_FILEPATH) as conn:
        if output_format == "python_list":
            cursor = conn.execute(
                query=sql_query,
                parameters={
                    "user_query": user_query,
                    "top_k": top_k,
                },
            )
            col_names: list[str] = [col_data[0] for col_data in cursor.description]
            rows: list[tuple] = cursor.fetchall()

            result = [
                dict(
                    zip(
                        col_names, 
                        row,
                    ),
                )
                for row in rows
            ]
        elif output_format == "polars":
            result = conn.sql(
                query=sql_query,
                params={
                    "user_query": user_query,
                    "top_k": top_k,
                },
            ).pl()

    return result

def semantic(
    user_query: str,
    top_k: int = 10,
    output_format: Literal["python_list", "polars"] = "python_list",
) -> list[dict] | pl.DataFrame:
    """
    Return `top_k` closest results to `user_query` using semantic search
    (over the semantic embedding vectors)
    """
    SUPPORTED_OUTPUT_FORMATS: Final[list[str]] = ["python_list", "polars"]
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format='{output_format}' is not currently supported" +
            "\nAvailable output formats are [" +
            ",".join([f"'{x}'" for x in SUPPORTED_OUTPUT_FORMATS]) +
            "]"
        )
    user_query_embedding: list[float] = embed_model.encode(user_query).tolist()
    sql_query: str = f"""
        SELECT      'semantic' AS 'search_method'
                ,   row_id
                ,   msg_text
                ,   ROW_NUMBER() OVER (ORDER BY similarity_score, RANDOM()) AS rank
        FROM    (
                    SELECT  row_id
                        ,   msg_text
                        ,   ARRAY_DISTANCE(
                                vector_msg_text, 
                                {user_query_embedding}::FLOAT[256]
                            ) AS similarity_score 
                    FROM    main
            )
        ORDER BY    similarity_score
        LIMIT       $top_k
        ;
    """
    with duckdb.connect(constants.DB_FILEPATH) as conn:
        if output_format == "python_list":
            cursor = conn.execute(
                query=sql_query,
                parameters={
                    "top_k": top_k,
                }
            )
            col_names: list[str] = [col_data[0] for col_data in cursor.description]
            rows: list[tuple] = cursor.fetchall()
            result = [
                dict(
                    zip(
                        col_names, 
                        row,
                    ),
                )
                for row in rows
            ]
        elif output_format == "polars":
            result = conn.sql(
                query=sql_query,
                params={
                    "top_k": top_k,
                }
            ).pl()

    return result

def hybrid_rrf(
    user_query: str,
    prefetch_k: int,
    top_k: int = 10,
    output_format: Literal["python_list", "polars"] = "python_list",
    high_rank_mitigation_constant: int = 60,
) -> list[dict] | pl.DataFrame:
    """
    Return `top_k` closest results to `user_query` using Reciprocal Rank Fusion hybrid search
    (combines both BM25 and semantic search)
    """
    SUPPORTED_OUTPUT_FORMATS: Final[list[str]] = ["python_list", "polars"]
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format='{output_format}' is not currently supported" +
            "\nAvailable output formats are [" +
            ",".join([f"'{x}'" for x in SUPPORTED_OUTPUT_FORMATS]) +
            "]"
        )
    bm25_df: pl.DataFrame = bm25(
        user_query=user_query,
        top_k=prefetch_k,
        output_format="polars",
    ).select(
        ["row_id", "msg_text", "rank",]
    ).rename(
        lambda colname: f"bm25_{colname}" if colname!="row_id" else colname
    )
    semantic_df: pl.DataFrame = semantic(
        user_query=user_query,
        top_k=prefetch_k,
        output_format="polars",
    ).select(
        ["row_id", "msg_text", "rank",]
    ).rename(
        lambda colname: f"semantic_{colname}" if colname!="row_id" else colname
    )

    if len(bm25_df) == 0:
        if output_format == "python_list":
            return semantic_df.to_dicts() 
        else:
            return semantic_df 

    combined_df: pl.DataFrame = bm25_df.join(
        semantic_df,
        on="row_id",
        how="full",
        validate="1:1",
    )
    assign_rank_to_missing: int = max(
        bm25_df.get_column("bm25_rank").max(),
        semantic_df.get_column("semantic_rank").max(),
    ) + 1
    combined_df = combined_df.with_columns(
        pl.col("bm25_rank").fill_null(assign_rank_to_missing),
        pl.col("semantic_rank").fill_null(assign_rank_to_missing),
    )
    combined_df = combined_df.with_columns(
        (
            ( 1 / (pl.col("bm25_rank") + high_rank_mitigation_constant) ) +
            ( 1 / (pl.col("semantic_rank") + high_rank_mitigation_constant) )
        ).alias("score")
    )
    combined_df = combined_df.top_k(top_k, by="score").sort("score", descending=True)
    combined_df = combined_df.with_row_index("rank", offset=1)
    combined_df = combined_df.with_columns(
        pl.lit("hybrid_rrf").alias("search_method")
    )
    combined_df = combined_df.with_columns(
        pl.col("bm25_msg_text").fill_null(pl.col("semantic_msg_text")).alias("msg_text")
    )
    combined_df = combined_df.select(
        ["search_method",
         "row_id",
         "msg_text",
         "score",
         "rank",
         ]
    )

    if output_format == "python_list":
        return combined_df.to_dicts() 

    elif output_format == "polars":
        return combined_df
