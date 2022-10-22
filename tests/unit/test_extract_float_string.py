from cleanlab_cli.dataset import extract_float_string

base_values = [180, 180.0, 180.5, 1.8e9]


class TestExtractFloatString:
    def test_base_value_strings(self):
        for v in base_values:
            assert extract_float_string(str(v)) == str(v)

    def test_percentage_strings(self):
        for v in base_values:
            assert extract_float_string(f"{v}%") == str(v)

    def test_dollar_strings(self):
        for v in base_values:
            assert extract_float_string(f"${v}") == str(v)

    def test_positive_sign_strings(self):
        for v in base_values:
            assert extract_float_string(f"+{v}") == str(v)

    def test_negative_sign_strings(self):
        for v in base_values:
            assert extract_float_string(f"-{v}") == f"-{v}"

    def test_positive_dollar_strings(self):
        for v in base_values:
            assert extract_float_string(f"+${v}") == str(v)

    def test_negative_dollar_strings(self):
        for v in base_values:
            assert extract_float_string(f"-${v}") == f"-{v}"

    def test_positive_percentage_strings(self):
        for v in base_values:
            assert extract_float_string(f"+{v}%") == str(v)

    def test_negative_percentage_strings(self):
        for v in base_values:
            assert extract_float_string(f"-{v}%") == f"-{v}"

    def test_invalid_strings(self):
        invalid_strings = ["2c", "2c2", "c2c", "c2"]
        for s in invalid_strings:
            assert extract_float_string(s) == ""
