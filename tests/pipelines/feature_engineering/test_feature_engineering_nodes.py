import logging

import pandas as pd

from nielsen_scope.pipelines.feature_engineering.nodes import build_feature_table


def test_build_feature_table_passes_the_frame_through_and_logs_its_shape(caplog):
    features = pd.DataFrame(
        {
            "isbn": ["9780000000001", "9780000000001", "9780000000002"],
            "end_date": pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-07"]),
            "volume": [1, 2, 3],
        }
    )
    with caplog.at_level(logging.INFO, logger="nielsen_scope.pipelines.feature_engineering.nodes"):
        out = build_feature_table(features)
    assert out is features
    assert "3 rows x 3 cols | ISBNs=2" in caplog.text