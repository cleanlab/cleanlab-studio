import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from cleanlab_studio.utils.data_enrichment.enrich import ClientEnrichmentProject
from cleanlab_studio.studio.enrichment import EnrichmentOptions, EnrichmentResults
import re

class TestClientEnrichmentProject(unittest.TestCase):
    def setUp(self):
        self.project = ClientEnrichmentProject("fake_api_key", "fake_id", "fake_name")

    @patch('cleanlab_studio.utils.data_enrichment.enrich.get_prompt_outputs')
    def test_online_inference_api_matches_run(self, mock_get_prompt_outputs):
        # Mock the get_prompt_outputs function
        mock_get_prompt_outputs.return_value = [
            {"trustworthiness_score": 0.9, "response": "positive"},
            {"trustworthiness_score": 0.8, "response": "negative"},
        ]

        # Create sample data and options
        data = pd.DataFrame({"text": ["This is great!", "This is terrible."]})
        options = EnrichmentOptions(
            prompt="Classify the sentiment: ${text}",
            constrain_outputs=["positive", "negative", "neutral"],
            optimize_prompt=True,
            quality_preset="medium",
            regex=None,
            tlm_options={}
        )
        new_column_name = "sentiment"

        # Call online_inference
        result = self.project.online_inference(data, options, new_column_name)

        # Assert that the result is an EnrichmentResults object
        self.assertIsInstance(result, EnrichmentResults)

        # Assert that the result DataFrame has the expected columns
        expected_columns = [
            "sentiment_trustworthiness_score",
            "sentiment",
            "sentiment_raw",
            "sentiment_log"
        ]
        self.assertListEqual(list(result.details().columns), expected_columns)

        # Assert that the run method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.project.run(options, new_column_name)

        # Assert that online_inference and run methods have the same signature
        import inspect
        online_inference_params = inspect.signature(self.project.online_inference).parameters
        run_params = inspect.signature(self.project.run).parameters

        self.assertEqual(set(online_inference_params.keys()), set(run_params.keys()) | {"data"})
        self.assertEqual(online_inference_params["options"].annotation, run_params["options"].annotation)
        self.assertEqual(online_inference_params["new_column_name"].annotation, run_params["new_column_name"].annotation)

if __name__ == '__main__':
    unittest.main()
