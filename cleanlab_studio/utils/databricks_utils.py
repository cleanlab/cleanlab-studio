import os

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit


def get_databricks_simple_imageset(path):
    """
    Returns a pandas DataFrame listing the classes and images from an image dataset hosted on DBFS.

    Similar to how pytorch expects a dataset to be structured, the dataset should be structured as follows:
    https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    https://help.cleanlab.ai/guide/concepts/datasets/#simple-zip
    ```
    [root]
    ├── class_1
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── class_2
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    └── ...
    ```

    Args:
        path: Path to image dataset in DBFS (ex: "dbfs:/FileStore/[...]/imageset-root").

    Returns:
        Pandas DataFrame with columns `class` and `row_data`. `class` is the class of the image. `row_data` is the image file.
    """

    # Initialize Spark Session
    spark = SparkSession.builder.appName("LoadAndCombineImages").getOrCreate()

    # List all subdirectories
    subdirs = [f.path for f in dbutils.fs.ls(path) if f.isDir()]

    # Function to load and label images from a subdir
    def load_and_label(subdir):
        label = os.path.normpath(subdir).split("/")[-1]
        return (
            spark.read.format("image")
            .load(subdir)
            .withColumn("class", lit(label))
            .withColumnRenamed("image", "row_data")
        )

    # Load and label all subdirectories, then reduce to combine
    combined_df = reduce(
        lambda df1, df2: df1.union(df2), (load_and_label(subdir) for subdir in subdirs)
    )

    return combined_df


def get_databricks_metadata_imageset(metadata_path):
    """
    Returns a pandas DataFrame, listing images and classes, from an image dataset stored on snowflake and described by a metadata file. The meta data file and directory storing it should match the description from https://help.cleanlab.ai/guide/concepts/datasets/#metadata-zip.

        Args:
            metadata_path: Path to metadata file in DBFS (ex: "dbfs:/FileStore/[...]/imageset-root/metadata.csv").

        Returns:
            Pandas DataFrame with columns `class` and `row_data`. `class` is the class of the image. `row_data` is the image file.
    """

    parent = lambda p: os.path.normpath(os.path.join(p, os.pardir))
    parent_of_parent = parent(parent(metadata_path))
    parent_of_parent = "" if parent_of_parent == "." else f"{parent_of_parent}/"

    # Initialize Spark Session
    spark = SparkSession.builder.appName("LoadAndCombineImages").getOrCreate()

    # Read metadata.csv under the path
    metadata = spark.read.csv(metadata_path, header=True).collect()

    # Initialize an empty DataFrame for images
    images_df = None

    # Load and label all images using the metadata dataframe
    for row in metadata:
        filepath, label = row["filepath"], row["label"]
        temp_df = (
            spark.read.format("image")
            .load(f"{parent_of_parent}{filepath}")
            .withColumn("class", lit(label))
        )
        images_df = temp_df if images_df is None else images_df.union(temp_df)

    return images_df
