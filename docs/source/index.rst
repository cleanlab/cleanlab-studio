.. Cleanlab Studio documentation master file, created by
   sphinx-quickstart on Wed May 10 18:40:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cleanlab Studio
***************

.. toctree::
   :maxdepth: 1
   :hidden:

   Quick Reference <quick_reference/index>
   Concepts <concepts/index>
..   API Reference <api/index> commented out since not yet useful (no docstrings)
..   FAQ <faq/index> commented out since not yet using


Cleanlab Studio automatically finds and fixes errors in your dataset to turn unreliable data into reliable ML models and insights.
This package provides a Python API and CLI to interact with the Cleanlab Studio app.

Quickstart
==========

Following this guide requires the following:

#. Python >= 3.8
#. A `Cleanlab Studio account <https://app.cleanlab.ai/>`_

Note that this guide utilizes the Python API for `cleanlab-studio`.
Usage of the command line interface can be found at :ref:`usage_cli_api`.

Installation
------------
You can install the Cleanlab Studio client `from PyPI <https://pypi.org/project/cleanlab-studio/>`_ with:

.. code-block:: bash

    pip install cleanlab-studio

If you already have the client installed and wish to upgrade to the latest version, run:

.. code-block:: bash

    pip install --upgrade cleanlab-studio

.. _quickstart_api_key:

Creating an API Key
-------------------

To upload datasets, create projects, and more, you must create an API key.

First, open your `account page in Cleanlab Studio <https://app.cleanlab.ai/account>`_.
If you're not already logged in, do so now.
Then, copy the API key from your account page.

You will use this API key in subsequent steps.

Uploading a Dataset
-------------------

The starting point for using Cleanlab Studio is uploading a dataset.

For more information on datasets in Cleanlab Studio, check out the following resources:

* :ref:`guide_dataset_modalities`
* :ref:`guide_dataset_formats`

For this guide, we will be using a character recognition dataset used in the `2023 Machine Hack DCAI competition <https://machinehack.com/tournaments/data_centric_ai_competition_2023>`_.
This dataset contains 9402 noisily labeled images.

The most common method of uploading a dataset to Cleanlab Studio through the Python API is to upload a Pandas or PySpark DataFrame.

Uploading this dataset is as simple as running the following commands in an interactive Python terminal (or a script):

.. code-block:: python

    >>> import cleanlab_studio
    >>> import pandas as pd
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> dcai_dataset_df = pd.read_csv(
        "https://cleanlab-public.s3.amazonaws.com/StudioDemoDatasets/dcai_external_media_dataset.csv"
    )
    >>> dcai_dataset_df.head(5)
       id                                                img label
    0   0  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     c
    1   1  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     h
    2   2  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     y
    3   3  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     p
    4   4  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     j
    >>>
    >>> studio.upload_dataset(dcai_dataset_df, "DCAI Demo Dataset")
    <your dataset ID>

Now, navigate to Cleanlab Studio. You should see a new dataset in your dashboard with the name "DCAI Demo Dataset".
Click `View Dataset`. You should see a page appear that looks like the following:

.. image:: /_images/dcai_dataset_screenshot.png
    :alt: Cleanlab Studio Dataset View


Creating a Project
------------------

After uploading a dataset, you can create a project. This, along with the other steps in this tutorial, can also be done in the web app.
Project creation will run a suite of checks on your dataset, including label error detection, outlier detection, and more.

Using your dataset ID from the previous step, you can create a project using the following command:

.. code-block:: python

    >>> import cleanlab_studio
    >>>
    >>> api_key: str = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> dataset_id: str = <your dataset ID>
    >>> studio.create_project(dataset_id, "DCAI Demo Project", "image")
    <your project ID>

This step will likely take approximately 15 minutes to run.
You can view progress in the `Cleanlab Studio Dashboard <https://app.cleanlab.ai/>`_ and you will also receive an email when the project is complete.

When the project completes, you can view the results by clicking on the project name.
You will see a page like the following, where you will be able to review your dataset and make corrections as needed:

.. image:: /_images/dcai_project_screenshot.png
    :alt: Cleanlab Studio Project View


Exporting your Results
----------------------
After you've made corrections to your dataset, you can export the results of your project to your local machine.

Results can either be exported by:

* Downloading Cleanlab Columns

  * a table containing metrics that Cleanlab Studio generated for your dataset
  * allows for further analysis of your dataset

* Applying Dataset Corrections

  * applies corrections made in your project to a local instance of your dataset
  * allows for training new models with your dataset

To export your results, you must first obtain the cleanset ID for your project.
Your cleanset ID can be found by clicking `Export Cleanset` on your cleanset page then `Export Using API`.

Using your cleanset ID, you can export your results as follows:

.. code-block:: python

    >>> import cleanlab_studio
    >>> import pandas as pd
    >>>
    >>> api_key = <your API key>
    >>> studio = cleanlab_studio.Studio(api_key)
    >>>
    >>> cleanset_id: str = <your cleanset ID>
    >>> dcai_cleanlab_cols: pd.DataFrame = studio.download_cleanlab_columns(cleanset_id)
    >>> dcai_cleanlab_cols.head(5)
        id  cleanlab_issue  cleanlab_label_quality cleanlab_suggested_label cleanlab_clean_label
    0   0           False                0.781765                     None                 None
    1   1            True                0.471000                        8                    8
    2   2           False                0.478483                        4                    4
    3   3           False                0.595736                     None                 None
    4   4           False                0.797456                     None                    i

    >>>
    >>> corrected_dcai_dataset_df: pd.DataFrame = studio.apply_corrections(
        cleanset_id,
        dcai_dataset_df,
    )
    >>> corrected_dcai_dataset_df.head(5)
       id                                                img label
    0   0  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     8
    1   1  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     h
    2   2  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     4
    3   3  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     p
    4   4  https://s.cleanlab.ai/DCA_Competition_2023_Dat...     i
