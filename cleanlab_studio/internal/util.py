import pathlib
from typing import Any, Optional, Tuple, TypeVar, Union, Callable, Dict, List
import math
import functools
import sys
import traceback
import contextlib
import subprocess
import platform
import re


import pandas as pd

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.errors import InvalidDatasetError, HandledError, ValidationError

try:
    import snowflake.snowpark as snowpark

    snowpark_exists = True
except ImportError:
    snowpark_exists = False

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from .dataset_source import (
    DatasetSource,
    FilepathDatasetSource,
    PandasDatasetSource,
)

dataset_source_types = [str, pathlib.Path, pd.DataFrame]
if pyspark_exists:
    dataset_source_types.append(pyspark.sql.DataFrame)
if snowpark_exists:
    dataset_source_types.append(snowpark.DataFrame)

DatasetSourceType = TypeVar("DatasetSourceType", bound=Union[tuple(dataset_source_types)])  # type: ignore


def init_dataset_source(
    dataset_source: DatasetSourceType, dataset_name: Optional[str] = None  # type: ignore
) -> DatasetSource:
    if isinstance(dataset_source, pd.DataFrame):
        if dataset_name is None:
            raise InvalidDatasetError("Must provide dataset name if uploading from a DataFrame")
        return PandasDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(
            filepath=pathlib.Path(dataset_source), dataset_name=dataset_name
        )
    elif snowpark_exists and isinstance(dataset_source, snowpark.DataFrame):
        from .dataset_source import SnowparkDatasetSource

        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return SnowparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif pyspark_exists and isinstance(dataset_source, pyspark.sql.DataFrame):
        from .dataset_source import PySparkDatasetSource

        if dataset_name is None:
            raise InvalidDatasetError("Must provide dataset name if uploading from a DataFrame")
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    else:
        raise InvalidDatasetError("Invalid dataset source provided")


def apply_corrections_snowpark_df(
    dataset: Any,
    cl_cols: Any,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> Any:
    from snowflake.snowpark.functions import (
        when,
        col,
        is_null,
        monotonically_increasing_id,
    )

    # to use lowercase column names, they need to be wrapped in double quotes to become valid identifiers
    # for example ("col" should be '"col"' so the engine will process the name as "col")
    # https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    label_col = quote(label_column)
    id_col = quote(id_col)
    action_col = quote("action")
    corrected_label_col = quote("corrected_label")
    cleanlab_final_label_col = quote("__cleanlab_final_label")

    corrected_ds = dataset
    session = dataset.session

    cl_cols_snowflake = session.create_dataframe(cl_cols)

    if id_col not in corrected_ds.columns:
        corrected_ds = corrected_ds.withColumn(id_col, monotonically_increasing_id())

    corrected_ds = (
        cl_cols_snowflake.select([id_col, action_col, corrected_label_col])
        .join(
            corrected_ds,
            on=id_col,
            how="left",
        )
        .withColumn(
            cleanlab_final_label_col,
            when(is_null(corrected_label_col), col(label_col)).otherwise(col(corrected_label_col)),
        )
        .drop(label_col, corrected_label_col)
        .withColumnRenamed(cleanlab_final_label_col, label_col)
    )

    corrected_ds = (
        corrected_ds.where((col(action_col) != "exclude") | is_null(col(action_col)))
        if not keep_excluded
        else corrected_ds
    ).drop(action_col)

    return corrected_ds


def apply_corrections_spark_df(
    dataset: Any,
    cl_cols: Any,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> Any:
    from pyspark.sql.functions import row_number, monotonically_increasing_id, when, col
    from pyspark.sql.window import Window

    corrected_ds_spark = dataset.alias("corrected_ds")
    if id_col not in corrected_ds_spark.columns:
        corrected_ds_spark = corrected_ds_spark.withColumn(
            id_col,
            row_number().over(Window.orderBy(monotonically_increasing_id())) - 1,
        )
    both = cl_cols.select([id_col, "action", "corrected_label"]).join(
        corrected_ds_spark.select([id_col, label_column]),
        on=id_col,
        how="left",
    )
    final = both.withColumn(
        "__cleanlab_final_label",
        when(col("corrected_label").isNull(), col(label_column)).otherwise(col("corrected_label")),
    )
    new_labels = final.select([id_col, "action", "__cleanlab_final_label"]).withColumnRenamed(
        "__cleanlab_final_label", label_column
    )

    res = corrected_ds_spark.drop(label_column).join(new_labels, on=id_col, how="right")
    res = (
        res.where((col("action").isNull()) | (col("action") != "exclude"))
        if not keep_excluded
        else res
    ).drop("action")

    return res


def apply_corrections_pd_df(
    dataset: pd.DataFrame,
    cl_cols: pd.DataFrame,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> pd.DataFrame:
    joined_ds: pd.DataFrame
    if id_col in dataset.columns:
        joined_ds = dataset.join(cl_cols.set_index(id_col), on=id_col)
    else:
        joined_ds = dataset.join(cl_cols.set_index(id_col).sort_values(by=id_col))
    joined_ds["__cleanlab_final_label"] = joined_ds["corrected_label"].fillna(dataset[label_column])

    corrected_ds: pd.DataFrame = dataset.copy()
    corrected_ds[label_column] = joined_ds["__cleanlab_final_label"]
    if not keep_excluded:
        corrected_ds = corrected_ds.loc[(joined_ds["action"] != "exclude").fillna(True)]
    else:
        corrected_ds["action"] = joined_ds["action"]
    return corrected_ds


def check_none(x: Any) -> bool:
    if isinstance(x, str):
        return x == "None" or x == "none" or x == "null" or x == "NULL"
    elif isinstance(x, float):
        return math.isnan(x)
    elif pd.isnull(x):
        return True
    else:
        return x is None


def check_not_none(x: Any) -> bool:
    return not check_none(x)


def telemetry(
    load_api_key: bool = False,
    track_all_frames: bool = True,
) -> Callable[..., Any]:
    """
    return a decorator to send user info and stack trace to backend if an exception is raised

    Args:
    load_api_key:
        if True, the decorator will try to load the api_key using CleanlabSettings.load()
        This is currently used to track the cli commands
        If False, then we assume we are tracking a method of the Studio class for the python app
        and we will load the api_key from the first arg of the function, will will be self of the Studio class
    track_all_frames:
        If True, return stack trace of all stack frames. Used for cli commands, because there is no
        end-user code running.
        If False, only return stack trace of cleanlab functions. Used for python app, so we don't track user code.
    """
    api_key: Optional[str] = None
    if load_api_key:
        with contextlib.suppress(Exception):
            api_key = CleanlabSettings.load().api_key

    def track(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to send stack trace to backend if an exception is raised

        Can only use on functions whose first arg is api_key
        """

        @functools.wraps(func)
        def tracked_func(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as err:
                with contextlib.suppress(Exception):
                    user_info = get_basic_info()
                    user_info["func_name"] = func.__name__
                    # if not load_api_key, then calling function from Studio class
                    # first arg is self, get _api_key from self
                    user_info["api_key"] = api_key if load_api_key else args[0]._api_key
                    trace_str = traceback.format_exc()
                    # only send stack trace for cleanlab functions
                    if track_all_frames:
                        cleanlab_traceback = trace_str
                    else:
                        # remove user code/local paths from trace
                        cleanlab_traceback = obfuscate_stack_trace(trace_str)

                    user_info["stack_trace"] = cleanlab_traceback
                    user_info["error_type"] = type(err).__name__
                    user_info["is_handled_error"] = isinstance(err, HandledError)
                    api.send_telemetry(user_info)
                raise err

        return tracked_func

    return track


def log_internal_error(error_message: str, stack_trace: str, api_key: Optional[str] = None) -> None:
    user_info = get_basic_info()
    user_info["api_key"] = api_key
    user_info["error_message"] = error_message
    user_info["stack_trace"] = obfuscate_stack_trace(stack_trace)
    api.send_telemetry(user_info)


def obfuscate_stack_trace(stack_trace: str) -> str:
    # remove stack frames for user code
    cleanlab_match = re.search("File.*cleanlab", stack_trace)
    cleanlab_traceback = stack_trace[cleanlab_match.start() :] if cleanlab_match else ""

    # clean up paths that don't contain "cleanlab-studio" which may contain local paths
    pattern1 = re.compile(r"File \"((?!cleanlab-studio).)*\n")
    cleanlab_traceback = pattern1.sub("File \n", cleanlab_traceback)

    # remove portions of paths preceding cleanlab-studio that may contain local paths
    pattern2 = re.compile(r"File([^\n]*?)cleanlab-studio")
    cleanlab_traceback = pattern2.sub('File "cleanlab-studio', cleanlab_traceback)
    return cleanlab_traceback


def get_basic_info() -> Dict[str, Any]:
    user_info: Dict[str, Any] = {}
    # get OS
    user_info["os"] = platform.system()
    user_info["os_release"] = platform.release()
    # get python version
    user_info["python_version"] = sys.version
    # get CLI version and dependencies
    studio_info = (
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                "cleanlab-studio",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )
    dependencies = []
    for line in studio_info:
        if line.startswith("Version:"):
            user_info["cli_version"] = line.split(" ")[1]
        elif line.startswith("Requires:"):
            dependencies = line.split(": ")[1].split(", ")
    all_packages = (
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8").split("\n")
    )
    package_versions = dict([package.split("==") for package in all_packages if "==" in package])
    user_info["dependencies"] = {
        dependency: package_versions.get(dependency) for dependency in dependencies
    }
    return user_info


def quote(s: str) -> str:
    return f'"{s}"'


def quote_list(l: List[str]) -> List[str]:
    return [quote(i) for i in l]

def get_prompt_outputs(studio, prompt, data, **kwargs):
    """Returns the outputs of the prompt for each row in the dataframe."""
    tlm = studio.TLM(**kwargs)
    formatted_prompts = data.apply(lambda x: prompt.format(**x), axis=1).to_list()
    outputs = tlm.try_prompt(formatted_prompts)
    return outputs

def extract_df_subset(
    df: pd.DataFrame, subset_indices: Union[Tuple[int, int], List[int], None]
) -> pd.DataFrame:
    """Extract a subset of the dataframe based on the provided indices. If no indices are provided, the entire dataframe is returned. Indices can be range or specific row indices."""
    if subset_indices is None:
        print("Processing your full dataset since `subset_indices` is None. This may take a while. Specify this argument to get faster results for a subset of your dataset.")
        return df
    if isinstance(subset_indices, range):
        subset_indices = subset_indices
    if isinstance(subset_indices, tuple):
        subset_indices = range(*subset_indices)
    subset_df = df.iloc[subset_indices].copy()
    return subset_df


def get_compiled_regex_list(regex: Union[str, re.Pattern, List[re.Pattern]]) -> List[re.Pattern]:
    """Compile the regex pattern(s) provided and return a list of compiled regex patterns."""
    if isinstance(regex, str):
        return [re.compile(rf"{regex}")]
    elif isinstance(regex, re.Pattern):
        return [regex]
    elif isinstance(regex, list):
        return regex
    else:
        raise ValidationError("Passed in regex can only be type one of: str, re.Pattern, or list of re.Pattern.")


def get_regex_match(response: str, regex_list: List[re.Pattern]) -> Union[str, None]:
    """Extract the first match from the response using the provided regex patterns. Return first match if multiple exist.
    Note: This function assumes the regex patterns each specify exactly 1 group that is the match group using '(<group>)'."""
    for regex_pattern in regex_list:
        pattern_match = regex_pattern.match(response)
        if pattern_match:
            return pattern_match.group(
                1
            )  # TODO: currently, this assumes 1 group in the supplied regex as the "match" group and takes that match if it exists.
    print(f'Your provided regex: {str(regex_list)} did not match a single LLM output across the dataset. This message is simply to inform you that the regex is not having any effect.')
    return None


def get_return_values_match(response: str, return_values_pattern: re.Pattern) -> Union[str, None]:
    """Extract the provided return values from the response using regex pattern. Return first extracted value if multiple exist."""
    exact_matches = re.findall(return_values_pattern, str(response))
    if len(exact_matches) > 0:
        return exact_matches[0]
    return None


def enrich_data(
    studio,
    prompt: str,
    data: pd.DataFrame,
    regex: Union[str, re.Pattern, List[re.Pattern]] = None,
    return_values: List[str] = None,
    subset_indices: Union[Tuple[int, int], List[int], None] = (0, 3),
    column_name_prefix: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    This method takes in a Studio client object, prompt template and a DataFrame and enriches the dataframe with the results of the prompting and associated trustworthiness scores.

    Args:
        prompt: Formatted f-string, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: {column_name}"
        regex: One or more expressions will be passed into re.compile or a list of already compiled regular expressions.
            If a list is proivded, the regexes are applied in order and first succesfull match is returned.
            **Note:** Regex patterns should each specify exactly 1 group that is the match group using parenthesis like so '.*(<desired match group pattern>)'.
            **Example:** `r'.*(Bird|[Rr]abbit).*'` will match any string that is the word 'Bird', 'Rabbit' or 'rabbit' into group 1.
        return_values: List of all possible values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
            After your regex is applied, there may be additional transformations applied to ensure the returned value is one of these.
        subset_indices: What subset of the supplied data to run this for. Can be either a list of unique indicies or a range. If None, we run on all the data.
        column_name_prefix: Optional prefix appended to all columns names that are returned.

    Returns:
        A DataFrame that now contains additional `metadata` and `trustworthiness` columns related to the prompt. Columns will have `column_name_prefix_` prepended to them if specified.
        `metadata` column = responses to the prompt and other data mutations if `regex` or `return_values` is not specified.
        `trustworthiness` column = trustworthiness of the prompt responses (which ignore the data mutations).
        **Note**: If any data mutations were made to the original response from the prompt, an additional `log` column will be added to the DataFrame that contains the raw output before the mutations were applied.
    """
    subset_data = extract_df_subset(data, subset_indices)
    outputs = get_prompt_outputs(studio, prompt, subset_data, **kwargs)

    if column_name_prefix != "":
        column_name_prefix = column_name_prefix + "_"

    if (
        regex is None and return_values is None
    ):  # we do not need to have a "logs" column as original output is not augmented by regex or return values
        subset_data[f"{column_name_prefix}metadata"] = [output["response"] for output in outputs]
        subset_data[f"{column_name_prefix}trustworthiness"] = [
            output["trustworthiness_score"] for output in outputs
        ]
        return subset_data

    subset_data[f"{column_name_prefix}logs"] = [output["response"] for output in outputs]
    subset_data[f"{column_name_prefix}trustworthiness"] = [
        output["trustworthiness_score"] for output in outputs
    ]

    if regex:
        regex_list = get_compiled_regex_list(regex)
        subset_data[f"{column_name_prefix}metadata"] = subset_data[
            f"{column_name_prefix}logs"
        ].apply(lambda x: get_regex_match(x, regex_list))
    else:
        subset_data[f"{column_name_prefix}metadata"] = subset_data[f"{column_name_prefix}logs"]

    if return_values:
        return_values_pattern = r"(" + "|".join(return_values) + ")"
        subset_data[f"{column_name_prefix}metadata"] = subset_data[
            f"{column_name_prefix}metadata"
        ].apply(lambda x: get_return_values_match(x, return_values_pattern))

    return subset_data

