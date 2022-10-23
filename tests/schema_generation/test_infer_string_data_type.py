import random
import uuid
from datetime import datetime, timedelta

from cleanlab_studio.cli.dataset import DataType, FeatureType
from cleanlab_studio.cli.dataset.schema_helpers import infer_types


class TestInferStringDataType:
    def test_infer_string_identifier(self):
        values = [uuid.uuid4().hex for _ in range(1000)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.string
        assert feature_type == FeatureType.identifier

    def test_infer_string_categorical(self):
        values = [["alpha", "beta", "gamma", "delta"][i % 4] for i in range(1000)]
        data_type, feature_type = infer_types(values)
        assert data_type == DataType.string
        assert feature_type == FeatureType.categorical

    def test_infer_string_datetime(self):
        datetime_now = datetime.now()
        datetime_values = [
            datetime_now
            + timedelta(
                days=random.randint(1, 100),
                hours=random.randint(1, 24),
                minutes=random.randint(1, 60),
                seconds=random.randint(1, 60),
            )
            for _ in range(1000)
        ]
        datetime_formats = ["%H:%M:%S", "%m/%d/%Y", "%d %b %Y", "%d %b %Y %H:%M:%S"]
        for datetime_format in datetime_formats:
            formatted_datetime_values = [x.strftime(datetime_format) for x in datetime_values]
            data_type, feature_type = infer_types(formatted_datetime_values)
            assert data_type == DataType.string
            assert feature_type == FeatureType.datetime

    def test_infer_string_text(self):
        lorem_ipsum = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore
        et dolore magna aliqua. Id diam maecenas ultricies mi eget mauris pharetra et. Mi sit amet mauris commodo
        quis imperdiet. Mi ipsum faucibus vitae aliquet nec ullamcorper. Urna cursus eget nunc scelerisque viverra
        mauris in aliquam. Pulvinar neque laoreet suspendisse interdum consectetur libero id faucibus nisl.
        Sollicitudin tempor id eu nisl nunc mi ipsum faucibus vitae.
        """
        lorem_len = len(lorem_ipsum)
        texts = []
        for _ in range(1000):
            start_idx = random.randint(1, lorem_len)
            end_idx = random.randint(start_idx, lorem_len)
            texts.append(lorem_ipsum[start_idx:end_idx])
        data_type, feature_type = infer_types(texts)
        assert data_type == DataType.string
        assert feature_type == FeatureType.text
