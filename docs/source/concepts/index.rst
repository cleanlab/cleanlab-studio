Concepts
********

Datasets
========

.. _concepts_dataset_modalities:

Modalities
----------
Cleanlab Studio supports datasets of the following modalities:

* Tabular
* Text
* Image

.. _concepts_dataset_formats:

Formats
-------

Text/Tabular
^^^^^^^^^^^^
Text and tabular datasets can be uploaded in the following formats:

CSV
"""
CSV is a standard format for text/tabular data, and most tools that process tabular data can export it as a CSV.
You should make sure your CSV file should be formatted as a standard CSV (e.g., use :code:`,` as the delimiter and :code:`"` as the quote character), which is usually the default.

The first row should contain headers naming all of the columns.
Furthermore, Cleanlab Studio requires that tabular data is non-jagged: each row should contain the same number of columns.
If some values are missing, those columns can be left blank (but the columns shouldn’t be missing entirely).

.. collapse:: Example dataset snippet
    :open:

    .. code-block::

        review_id,review,label
        f3ac,The sales rep was fantastic!,positive
        d7c4,He was a bit wishy-washy.,negative
        439a,"They kept using the word ""obviously,"" which was off-putting.",positive
        a53f,,negative

JSON
""""

JSON is a standard data interchange format.
Cleanlab Studio expects text/tabular data encoded as a JSON array of JSON objects, where each object has the same set of keys.
Values must be primitives: Cleanlab Studio doesn’t support nested JSON structures, you must flatten them.

All objects in your dataset must contain the same set of keys; if a value is missing from one of your rows of data, map it to :code:`null`.

.. collapse:: Example dataset snippet
    :open:

    .. code-block:: json

        [
            {
                "review_id": "f3ac",
                "review": "The sales rep was fantastic!",
                "label": "positive"
            },
            {
                "review_id": "d7c4",
                "review": "He was a bit wishy-washy.",
                "label": "negative"
            },
            {
                "review_id": "439a",
                "review": "They kept using the word \"obviously,\" which was off-putting.",
                "label": "positive"
            },
            {
                "review_id": "a53f",
                "review": null,
                "label": "negative"
            }
        ]

Excel
"""""

Cleanlab Studio supports Microsoft Excel files.
The first sheet will be imported as your dataset. The first row of your sheet should contain names for all of the columns.

.. collapse:: Example dataset snippet
    :open:

    = ========= ================================================== =========
    _ review_id                                             review     label
    = ========= ================================================== =========
    0      f3ac                       The sales rep was fantastic!  positive
    1      d7c4                          He was a bit wishy-washy.  negative
    2      439a  They kept using the word "obviously," which wa...  positive
    3      a53f                                                     negative
    = ========= ================================================== =========

Pandas/PySpark DataFrame
""""""""""""""""""""""""

Cleanlab Studio’s Python API supports a number of DataFrame formats, including Pandas DataFrames and PySpark DataFrames.
You can upload directly from a DataFrame in a Python script or Jupyter notebook.

.. collapse:: Example dataset snippet
    :open:

    = ========= ================================================== =========
    _ review_id                                             review     label
    = ========= ================================================== =========
    0      f3ac                       The sales rep was fantastic!  positive
    1      d7c4                          He was a bit wishy-washy.  negative
    2      439a  They kept using the word "obviously," which wa...  positive
    3      a53f                                                     negative
    = ========= ================================================== =========

Image
^^^^^
Image datasets can be uploaded in the following formats:

Simple ZIP
""""""""""
Images can be uploaded in ZIP file format, with a folder for each class and image files in each folder.
The folder names are interpreted as class labels.

.. image:: /_images/simple_zip_folder.png
    :alt: Simple ZIP Folder Layout
    :height: 540px
    :align: center

Metadata ZIP
""""""""""""
Images can be uploaded in ZIP file format, with a CSV manifest.
This manifest, which must be named :code:`metadata.csv` and placed at the top-level of the zipped directory, contains mappings between relative filepaths and labels.
The images in the ZIP can be in an arbitrary layout.

The metadata file must be formatted as a standard CSV (e.g., use :code:`,` as the delimiter and :code:`"` as the quote character).

.. image:: /_images/metadata_zip_folder.png
    :alt: Metadata ZIP Folder Layout
    :height: 540px
    :align: center

External Media
""""""""""""""
Images can be supplied using public URLs in any of our supported tabular formats (CSV, JSON, XLS/XLSX, DataFrame). If using a CSV, ensure that it is formatted as a standard CSV (e.g., use :code:`,` as the delimiter and :code:`"` as the quote character).
If using JSON, ensure that it is encoded as a JSON array of JSON objects, where each object has the same set of keys.
Values must be primitives: Cleanlab Studio doesn’t support nested JSON structures, you must flatten them.

One of your columns should contain a sequence of URLs, each pointing to a single hosted image.
These URLs must be either public or pre-signed; no additional authentication can be required to access the images.
Your dataset can contain arbitrary other columns, in addition to the image and label columns.

.. collapse:: Example dataset snippet
    :open:

    =  ==  ================================================= =====
    _  id                                                img label
    =  ==  ================================================= =====
    0   0  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     c
    1   1  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     h
    2   2  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     y
    3   3  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     p
    4   4  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     j
    =  ==  ================================================= =====

Schemas
-------
Schemas are used to define the data and feature types of the fields in your uploaded datasets.
While Cleanlab Studio is generally quite accurate with inferring these types, sometimes you may want to override our inferences.

Cleanlab Studio supports the following data and feature types:

=========   =================================================
Data type   Feature type
=========   =================================================
string      text, categorical, datetime, identifier, filepath
integer     categorical, datetime, identifier, numeric
float       datetime, numeric
boolean     boolean
=========   =================================================

In the case that you want to override the inferred schema, you can pass in a schema override.
The format of schema overrides are as follows:

.. code-block::

    {
        "<column_name>": {
            "data_type": "<override_data_type>",
            "feature_type": "<override_feature_type>",
        },
        ...
    }

Projects
========

Machine Learning Tasks
----------------------
Cleanlab Studio supports the following ML tasks:

* Multi-class classification (:code:`multi-class`)
* Multi-label classification (:code:`multi-label`) **ALERT: make sure this is released before the docs**


Modality
--------
Cleanlab Studio supports the following project modalities:

* Text
* Tabular
* Image

Model Type
----------
Cleanlab Studio supports the following model types:

* fast
* regular
