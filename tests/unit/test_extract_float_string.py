from cleanlab_cli.dataset import extract_float_string


class TestExtractFloatString:
    def test_percentage_string(self):
        assert extract_float_string("180.5%") == "180.5"

    def test_dollar_string(self):
        assert extract_float_string("$180.5") == "180.5"

    def test_float_string(self):
        assert extract_float_string("180.5") == "180.5"

    def test_invalid_string(self):
        assert extract_float_string("c2ab3") == ""
