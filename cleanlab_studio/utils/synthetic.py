"""
Collection of utility functions for Cleanlab Studio Python API
"""

from typing import Dict, Optional, Tuple, cast
import pandas as pd


_near_duplicate_id_column = "near_duplicate_cluster_id"


class _SyntheticDatasetScorer:
    """
    Computes the issue scores for a dataset consisting of real and synthetic data, to evaluate any overarching issues within the synthetic dataset.

    Args:
        cleanset_df: The dataframe containing the dataset to score. It should contain a column named "real_or_synthetic" that indicates whether each example is real or synthetic.
        real_or_synth_column: The name of the column that indicates whether each example is real or synthetic.
        synthetic_class_names: A tuple containing the class names of the "real_or_synthetic" column (ie. how to identify the examples that are real and synthetic).
    """

    def __init__(
        self,
        cleanset_df: pd.DataFrame,
        real_or_synth_column: str,
        synthetic_class_names: Tuple[str, str],
    ):
        self.cleanset_df = cleanset_df
        self.synthetic_type, self.real_type = synthetic_class_names
        self.real_or_synth_column = real_or_synth_column

    def _display_example_counts(self) -> None:
        """Displays the number of real and synthetic
        examples in the dataset.
        """
        column_name = self.real_or_synth_column
        real_target = self.real_type
        query = f"{column_name} == '{real_target}'"
        num_real_examples = len(self.cleanset_df.query(query))
        num_synthetic_examples = len(self.cleanset_df) - num_real_examples
        print(f"Number of real examples: {num_real_examples}")
        print(f"Number of synthetic examples: {num_synthetic_examples}")

    def score_synthetic_dataset(self) -> Dict[str, float]:
        """
        Computes the issue scores for a dataset consisting of real and synthetic data, to evaluate any overarching issues within the synthetic dataset.
        """
        self._display_example_counts()
        scores = {
            "unrealistic": self._unrealistic_score(),
            "unrepresentative": self._unrepresentative_score(),
            "unvaried": self._unvaried_score(),
            "unoriginal": self._unoriginal_score(),
        }
        return scores

    def _unrealistic_score(self) -> float:
        """
        Evaluates how distinguishable the synthetic data appears from real data. High values indicate there are many unrealistic-looking synthetic samples which look obviously fake.
        """
        return self._lis_to_synthetic_score(self.synthetic_type)

    def _unrepresentative_score(self) -> float:
        """
        Evaluates how poorly represented the real data is amongst the synthetic data samples. High values indicate there exist tails of the real data distribution (rare scenarios) that the distribution of synthetic samples fails to capture.
        """

        return self._lis_to_synthetic_score(self.real_type)

    def _unvaried_score(self) -> float:
        """
        Evaluates the diversity among synthetic samples. High values indicate a non-diverse synthetic data generator that produces many similar samples which look like near duplicates of other synthetic samples.
        """
        target_type = self.synthetic_type
        return self._calculate_synthetic_duplicate_scores(target_type, False)

    def _unoriginal_score(self) -> float:
        """
        Evaluates the lack of novelty of the synthetic data. High values indicate many synthetic samples are near duplicates of an example from the real dataset, i.e. the synthetic data generator may be memorizing the real data too closely and failing to generalize.
        """
        return self._calculate_synthetic_duplicate_scores(
            self.synthetic_type,
            True,
        )

    def _lis_to_synthetic_score(self, target_type: str) -> float:
        """
        Computes the synthetic score based on the label quality scores of the dataset.

        Args:
            target_type: The type of examples to compute the score for. If the target type is "synthetic", or a class name that corresponds to synthetic examples, then the score is computed based on the label issue scores of the synthetic examples. If the target type is "real", or a class name that corresponds to real examples, then the score is computed based on the label issue scores of the real examples.

        Returns:
            The synthetic score, which is the mean label issue score of the target type.
        """
        query = f"{self.real_or_synth_column} == @target_type"
        filtered_df = self.cleanset_df.query(query)
        mean_label_issue_score = filtered_df["label_issue_score"].mean()
        score = 1 - mean_label_issue_score
        return cast(float, score)

    def _calculate_synthetic_duplicate_scores(
        self,
        target_type: str,
        contains_real: bool,
    ) -> float:
        """
        Computes the synthetic duplicate score.
        """
        # Real or synthetic column
        _rs_column = self.real_or_synth_column

        # Synthetic class names (real or synthetic)
        class_names = self.cleanset_df[_rs_column].unique()
        assert len(class_names) == 2, "The dataset should contain both real and synthetic examples"
        target_mask = class_names == target_type
        complementary_class: str = class_names[~target_mask][0]

        df_near_duplicates = self.cleanset_df.query("is_near_duplicate")
        if df_near_duplicates.empty:
            return 0.0

        group_contains_real = df_near_duplicates.groupby(_near_duplicate_id_column).apply(
            lambda group: (group[_rs_column] == complementary_class).any()
        )
        filtered_df = df_near_duplicates[df_near_duplicates[_rs_column] == target_type]

        if contains_real:
            synthetic_duplicate_count = filtered_df[
                filtered_df[_near_duplicate_id_column].map(group_contains_real)  # noqa: E501
            ].shape[0]
        else:
            synthetic_duplicate_count = filtered_df[
                ~filtered_df[_near_duplicate_id_column].map(group_contains_real)  # noqa: E501
            ].shape[0]

        total_synthetic_examples = self.cleanset_df[
            self.cleanset_df[self.real_or_synth_column] == target_type
        ].shape[0]
        score = synthetic_duplicate_count / total_synthetic_examples
        return cast(float, score)


def score_synthetic_dataset(
    cleanset_df: pd.DataFrame,
    real_or_synth_column: str = "real_or_synthetic",
    synthetic_class_names: Optional[Tuple[str, str]] = None,
) -> Dict[str, float]:
    """
    Computes the issue scores for a dataset consisting of real and synthetic data, to evaluate any overarching issues within the synthetic dataset.

    Args:
        cleanset_df: The dataframe containing the dataset to score. It should contain a column named "real_or_synthetic" that indicates whether each example is real or synthetic. It should also have the cleanset columns provided by Cleanlab Studio.
        real_or_synth_column: The name of the column that indicates whether each example is real or synthetic.
        synthetic_class_names: The class names of the "real_or_synthetic" column (ie. which class corresponds to real examples, which to synthetic examples). If None, the default values are ("synthetic", "real"). The first class name should correspond to the synthetic examples, and the second class name should correspond to the real examples.
    """

    # Configure scorer
    scorer = _SyntheticDatasetScorer(
        cleanset_df=cleanset_df,
        synthetic_class_names=synthetic_class_names or ("synthetic", "real"),
        real_or_synth_column=real_or_synth_column,
    )
    return scorer.score_synthetic_dataset()
