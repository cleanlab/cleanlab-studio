from cleanlab_studio.cli.dataset import DataType, FeatureType
from cleanlab_studio.cli.dataset.schema_helpers import infer_types


class TestInferBooleanDataType:
    def test_infer_boolean(self):
        combos = [
            [True, False],
            # ['yes', 'no'],
            # ['t', 'f'],
            # [1, 0],
            # [1.0, 0.0]
        ]
        # we do not look out for these non-boolean cases explicitly at the moment
        for combo in combos:
            values = [combo[i % 2] for i in range(1000)]
            data_type, feature_type = infer_types(values)
            assert data_type == DataType.boolean
            assert feature_type == FeatureType.boolean
