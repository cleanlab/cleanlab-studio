.. _guide_python_api:

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

Get Cleanset ID
---------------
You can get the ID of the latest cleanset for your project given the project ID returned from the create project step.

.. code-block:: python

    >>> import cleanlab_studio
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>> project_id = <your project ID>
    >>>
    >>> studio.get_latest_cleanset_id(project_id)
    <cleanset ID>


Get Cleanset Status
-------------------
You won't be able to export your cleanset or make any corrections to your project until the cleanset is ready.
You can poll for your cleanset status using your cleanset ID. This function will block until the cleanset is ready or there is an error.
You can optionally specify a :code:`timeout` parameter to define the number of seconds to block before returning.
The function will return :code:`True` if the cleanset is ready and :code:`False` if there was a cleanset error or :code:`timeout` expired.

.. code-block:: python

    >>> import cleanlab_studio
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>> cleanset_id = <your cleanset ID>
    >>>
    >>> studio.poll_cleanset_status(cleanset_id)
    Cleanset Progress: | Step 0/5, Initializing...

Export
======
Once your cleanset is ready, there are a couple ways you can export it.

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
After making corrections to your dataset in Cleanlab Studio, you can apply them to a local copy of your dataset in dataframe form, given the cleanset ID.
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
