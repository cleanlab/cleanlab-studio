.. _quick_ref_python_api:

Python API
**********

Upload Dataset
==============

Upload File
-----------
If you have a dataset saved to your filesystem in one of Cleanlab Studio's :ref:`supported formats<concepts_dataset_formats>`, you can upload it from a filepath.

.. code-block:: python

    >>> import cleanlab_studio
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> dataset_path: str = <path to your dataset>
    >>> studio.upload_dataset(dataset_path)
    <dataset ID>


Upload DataFrame
----------------
If you have a dataset in a :code:`Pandas` or :code:`PySpark` DataFrame, you can upload it directly.

.. code-block:: python

    >>> import cleanlab_studio
    >>> import pandas as pd
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> dataset_df: pd.DataFrame = <your dataset>
    >>> dataset_name: str = <your_dataset_name>
    >>> studio.upload_dataset(dataset_df, dataset_name)
    <dataset ID>


Create Project
==============
You can create a project for a dataset given the dataset ID returned from the upload step.

.. code-block:: python

    >>> import cleanlab_studio
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> project_name: str = <your project name>
    >>> modality: str = <your project modality>
    >>> studio.create_project(dataset_id, project_name, modality)
    <project ID>

Export
======

Download Cleanlab Columns
-------------------------
You can download the cleanlab columns from your project given the cleanset ID.
This will return a dataframe containing per-row information computed on your dataset, along with corrections you've made.

.. code-block:: python

    >>> import cleanlab_studio
    >>> import pandas as pd
    >>>
    >>> api_key = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> cleanset_id: str = <your cleanset ID>
    >>> cleanlab_cols: pd.DataFrame = studio.download_cleanlab_columns(cleanset_id)


Apply Corrections
-----------------
You can apply the corrections from your project given a copy of your dataset in dataframe form and the cleanset ID.
This will yield a dataframe of the corrections applied to your dataset.

.. code-block:: python

    >>> import cleanlab_studio
    >>> import pandas as pd
    >>>
    >>> api_key = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> dataset_df: pd.DataFrame = <your dataset>
    >>> cleanset_id: str = <your cleanset ID>
    >>> corrected_df: pd.DataFrame = studio.apply_corrections(cleanset_id, dataset_df)
