.. _guide_cli_api:

CLI
***

Upload Dataset
==============

Upload File
-----------
If you have a dataset saved to your filesystem in one of Cleanlab Studio's :ref:`supported formats<concepts_dataset_formats>`, you can upload it from a filepath.

.. code-block:: bash

    cleanlab dataset upload -f <path to your dataset>


Export
======

Download Cleanlab Columns
-------------------------
You can download the cleanlab columns from your project given the cleanset ID.
This will yield a CSV containing per-row information computed on your dataset, along with corrections you've made.

.. code-block:: bash

    cleanlab cleanset download --id <cleanset ID> --output <path to save to> --all


Apply Corrections
-----------------
You can apply the corrections from your project given a copy of your dataset in file form and the cleanset ID.
This will yield an output file of the corrections applied to your dataset.

.. code-block:: bash

    cleanlab cleanset download --id <cleanset ID> --filepath <path to your dataset> --output <path to save to> --all
