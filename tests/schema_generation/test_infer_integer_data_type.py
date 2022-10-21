import random
from typing import List, Any

from cleanlab_cli.dataset import DataType, FeatureType
from cleanlab_cli.dataset.schema_helpers import infer_types


def cast_to_strings(values: List[Any]):
    return [str(x) for x in values]


class TestInferIntegerDataType:
    def test_infer_integer_identifier(self):
        values = [i for i in range(1000)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.integer
        assert feature_type == FeatureType.identifier

        data_type, feature_type = infer_types(cast_to_strings(values))
        assert data_type == DataType.integer
        assert feature_type == FeatureType.identifier

    def test_infer_integer_categorical(self):
        values = [i % 10 for i in range(1000)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.integer
        assert feature_type == FeatureType.categorical

        data_type, feature_type = infer_types(cast_to_strings(values))
        assert data_type == DataType.integer
        assert feature_type == FeatureType.categorical

    def test_infer_integer_numeric(self):
        random.seed(43)
        values = [random.randrange(0, 100) for _ in range(100)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.integer
        assert feature_type == FeatureType.numeric

        data_type, feature_type = infer_types(cast_to_strings(values))
        assert data_type == DataType.integer
        assert feature_type == FeatureType.numeric
