import pytest
import numpy as np

from cleanlab_studio.utils.synthetic import score_synthetic_dataset
from dataframe import load_cleanset_df


@pytest.fixture
def df():
    return load_cleanset_df()


@pytest.fixture
def synthetic_scores(df):
    return score_synthetic_dataset(df)


def test_scores_are_in_a_dict(synthetic_scores):
    assert isinstance(synthetic_scores, dict)
    assert all(isinstance(k, str) for k in synthetic_scores.keys())
    assert all(isinstance(v, float) for v in synthetic_scores.values())
    assert len(synthetic_scores) == 4


def test_scores_lie_between_0_and_1(synthetic_scores):
    assert all(0 <= v <= 1 for v in synthetic_scores.values())


def test_scores_work_for_dataframes_without_near_duplicates(synthetic_scores, df):
    # Set "near_duplcicate_cluster_id" to nan to simulate a dataframe without near duplicates
    df_copy = df.copy()
    df_copy["is_near_duplicate"] = False
    df_copy["near_duplicate_cluster_id"] = float("nan")

    # Expect "unrealistic", "unoriginal" and "unvaried" scores to be 0.0
    scores_of_copy = score_synthetic_dataset(df_copy)
    affected_scores = ["unoriginal", "unvaried"]
    assert all(scores_of_copy[k] == 0.0 for k in affected_scores)

    # The other scores are not affected by the absence of near duplicates
    assert all(
        synthetic_scores[k] == scores_of_copy[k]
        for k in synthetic_scores.keys()
        if k not in affected_scores
    )


@pytest.mark.parametrize("target_type", ["synthetic", "real"])
def test_scores_raise_error_with_only_real_or_synthetic_data(target_type, df):
    # Set "real_or_synthetic" to "real" to simulate a dataframe with only real data
    df_copy = df.copy()
    df_copy["real_or_synthetic"] = target_type

    with pytest.raises(AssertionError) as excinfo:
        score_synthetic_dataset(df_copy)

    assert "The dataset should contain both real and synthetic examples" in str(excinfo.value)


def test_scores_work_with_alternative_synthetic_class_names(synthetic_scores, df):
    original_synthetic_class_names = ["synthetic", "real"]
    new_synthetic_class_names = ["artificial", "original"]

    # Map the original synthetic class names to the new synthetic class names
    class_mapping = dict(zip(original_synthetic_class_names, new_synthetic_class_names))

    df_copy = df.copy()
    df_copy["real_or_synthetic"] = df_copy["real_or_synthetic"].replace(class_mapping)

    new_scores = score_synthetic_dataset(df_copy, synthetic_class_names=new_synthetic_class_names)
    assert new_scores == synthetic_scores


def test_scores_are_consistent_across_ordering(df, synthetic_scores):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac=1)

    new_scores = score_synthetic_dataset(df_copy)
    np.testing.assert_allclose(
        list(new_scores.values()),
        list(synthetic_scores.values()),
        atol=1e-6,
        err_msg="Shuffling the dataframe should not affect the scores",
    )
