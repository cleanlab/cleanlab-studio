from cleanlab_studio.dataset import DataType, FeatureType
from cleanlab_studio.dataset.schema_helpers import infer_types
from tests.schema_generation.utils import cast_to_strings


class TestInferFloatDataType:
    def test_infer_float_numeric(self):
        values = [float(i**2) for i in range(1000)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.float
        assert feature_type == FeatureType.numeric

        data_type, feature_type = infer_types(cast_to_strings(values))
        assert data_type == DataType.float
        assert feature_type == FeatureType.numeric
