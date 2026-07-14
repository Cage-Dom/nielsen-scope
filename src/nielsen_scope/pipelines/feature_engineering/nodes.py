import logging
import pandas as pd

logger = logging.getLogger(__name__)

def build_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    """Persists the SQL feature engineering step via the 
    feature_table"""

    logger.info(
        "sales_features: %d rows x %d cols | ISBNs=%d | weeks %s..%s",
        len(features),
        features.shape[1],
        features["isbn"].nunique(),
        features["end_date"].min(),
        features["end_date"].max(),
    )
    return features