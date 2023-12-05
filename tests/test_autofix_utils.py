import pandas as pd
import pytest
from cleanlab_studio.internal.util import (
    get_param_values,
    _update_label_based_on_confidence,
    _get_top_fraction_ids,
    _get_indices_to_drop
)
import numpy as np


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
    def test_get_param_values(self, strategy, expected_results):
        cleanlab_columns = pd.DataFrame()
        cleanlab_columns["is_label_issue"] = [True] * 3 + [False] * 7
        cleanlab_columns["is_near_duplicate"] = [True] * 6 + [False] * 4
        cleanlab_columns["is_outlier"] = [True] * 6 + [False] * 4
        cleanlab_columns["is_ambiguous"] = [True] * 10

        params = get_param_values(cleanlab_columns, None, strategy)
        assert params == expected_results

    @pytest.mark.parametrize(
        "row, expected_updated_row",
        [
            (
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.6,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.6,
                    "label": "label_1",
                    "suggested_label": "label_1",
                    "is_issue": False,
                },
            ),
            (
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.5,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.5,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
            ),
            (
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.4,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
                {
                    "is_label_issue": True,
                    "suggested_label_confidence_score": 0.4,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
            ),
            (
                {
                    "is_label_issue": False,
                    "suggested_label_confidence_score": 0.4,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
                {
                    "is_label_issue": False,
                    "suggested_label_confidence_score": 0.4,
                    "label": "label_0",
                    "suggested_label": "label_1",
                    "is_issue": True,
                },
            ),
        ],
        ids=[
            "is a label issue with confidence score greater than threshold",
            "is a label issue with confidence score equal to threshold",
            "is a label issue with confidence score less than threshold",
            "is not a label issue",
        ],
    )
    def test_update_label_based_on_confidence(self, row, expected_updated_row):
        conf_threshold = 0.5
        updated_row = _update_label_based_on_confidence(row, conf_threshold)
        assert updated_row == expected_updated_row

    def test_get_top_fraction_ids(self):
        cleanlab_columns = pd.DataFrame()

        cleanlab_columns["cleanlab_row_ID"] = np.arange(10)
        cleanlab_columns["is_dummy"] = [False] * 5 + [True] * 5
        cleanlab_columns["dummy_score"] = np.arange(10) * 0.1
        top_ids = _get_top_fraction_ids(cleanlab_columns, "dummy", 3)
        assert set(top_ids) == set([5, 6, 7])

    def test_get_top_fraction_ids_near_duplicate(self):
        cleanlab_columns = pd.DataFrame()

        cleanlab_columns["cleanlab_row_ID"] = np.arange(12)
        cleanlab_columns["is_near_duplicate"] = [False] * 6 + [True] * 6
        cleanlab_columns["near_duplicate_score"] = np.arange(12) * 0.1
        cleanlab_columns["near_duplicate_cluster_id"] = [None] * 6 + [0, 0, 1, 1, 1, 1]

        top_ids = _get_top_fraction_ids(cleanlab_columns, "near_duplicate", 5)
        assert set(top_ids) == set([6, 8, 10])