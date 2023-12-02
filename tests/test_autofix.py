import pandas as pd
import pytest
from cleanlab_studio.internal.util import get_autofix_defaults


class TestAutofix:
    @pytest.mark.parametrize(
        "strategy, expected_results",
        [
            (
                "optimized_training_data",
                {
                    "drop_ambiguous": 0,
                    "drop_label_issue": 2,
                    "drop_near_duplicate": 2,
                    "drop_outlier": 3,
                    "relabel_confidence_threshold": 0.95,
                },
            ),
            (
                "drop_all_issues",
                {
                    "drop_ambiguous": 10,
                    "drop_label_issue": 3,
                    "drop_near_duplicate": 6,
                    "drop_outlier": 6,
                },
            ),
            (
                "suggested_actions",
                {
                    "drop_near_duplicate": 6,
                    "drop_outlier": 6,
                    "relabel_confidence_threshold": 0.0,
                },
            ),
        ],
        ids=["optimized_training_data", "drop_all_issues", "suggested_actions"],
    )
    def test_get_autofix_defaults(self, strategy, expected_results):
        cleanlab_columns = pd.DataFrame()
        cleanlab_columns["is_label_issue"] = [True] * 3 + [False] * 7
        cleanlab_columns["is_near_duplicate"] = [True] * 6 + [False] * 4
        cleanlab_columns["is_outlier"] = [True] * 6 + [False] * 4
        cleanlab_columns["is_ambiguous"] = [True] * 10

        params = get_autofix_defaults(cleanlab_columns, strategy)
        assert params == expected_results
