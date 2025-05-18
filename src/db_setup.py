import duckdb
import numpy as np
import polars as pl

from src import constants, loggers
from src.language_models import embed_model

logger = loggers.get_logger(__name__)

logger.info(f"Reading in data from {constants.INPUT_DATA_FILEPATH}")
labels: list[str] = []
message_texts: list[str] = []
with open(constants.INPUT_DATA_FILEPATH, "r") as file:
    for line in file.readlines():
        label, message_text = line.split("\t", 1)
        labels.append(label.strip())
        message_texts.append(message_text.strip())
main_df = pl.DataFrame({"label": labels, "msg_text": message_texts})

logger.info("creating semantic embeddings")
semantic_vectors_msg_text: np.ndarray = embed_model.encode(
    main_df.get_column("msg_text").to_list()
)
main_df = main_df.with_columns(
    pl.Series(
        "vector_msg_text",
        semantic_vectors_msg_text,
    )
)

with duckdb.connect(constants.DB_FILEPATH) as con:
    logger.info("Installing vector similarity search (VSS) extension")
    con.execute(
        """
        -- install Vector Similarity Search extension
        INSTALL vss;
        LOAD vss;
        SET hnsw_enable_experimental_persistence = true;
        """
    )
    logger.info(f"Writing to local database {constants.DB_FILEPATH}")
    con.execute("DROP TABLE IF EXISTS main;")
    con.sql(
        """
        CREATE TABLE    main
        AS
        SELECT  UUID() AS row_id
            ,   * 
        FROM    main_df 
        ;
    """
    )
    logger.info("Creating HNSW semantic vector index")
    con.execute(
        """
        CREATE INDEX    semantic_vec_hnsw_index
        ON              main 
        USING           HNSW (vector_msg_text)
        WITH            (metric = 'cosine')
        ;
        """
    )
    logger.info("Creating Full Text Search (FTS) index")
    create_fts_index_query: str = """
        PRAGMA CREATE_FTS_INDEX(
            'main',         -- table name
            'row_id',       -- doc identifier column
            'msg_text',     -- column(s) to index
            overwrite=1     -- overwrite existing index
        );
        """.strip()
    logger.info("\n" + create_fts_index_query)
    con.execute(create_fts_index_query)
    logger.warning("FTS index does not automatically update when new data is inserted")
