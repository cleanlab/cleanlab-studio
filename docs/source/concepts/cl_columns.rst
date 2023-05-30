.. _concepts_cl_columns:

Cleanlab Columns
****************

Introduction
============

Cleanlab Studio will run a number of analyses on your dataset to help identify potential data issues. Currently we look for the following issues in datasets:
- Label issues
- Outliers
We will also continue to add new analyses soon. The raw outputs of each analysis will be available to download as Cleanlab columns by calling ``studio.download_cleanlab_columns()``.

For each analysis we will often expose two columns, one being a numeric score between 0 and 1 indicating the quality of a sample for that analysis (lower score means lower quality), and the other being a boolean, which indicates whether the sample is likely to have an issue or not. The different analyses are run separately, and hence they should be independently. For example, samples with label issues are not neccessarily outliers, and vice versa, and it is possible to have a sample with label issue marked as ``False``, but outlier marked as ``True``.

We give more details on the different Cleanlab columns below.

Label Issues
============

Cleanlab Label quality
-------------
The ``cleanlab_label_quality`` column contains a score bounded between 0 and 1. The score is calculated using confident learning. The lower the score of a sample, the more likely it has a label issue.

Cleanlab Issue
-----
The ``cleanlab_issue`` column contains a boolean value, with ``True`` indicating that the sample is likely to have a label issue. The value is obtained by thresholding the ``label_quality`` score with confident learning.

Cleanlab Suggested label
---------------
The ``cleanlab_suggested_label`` column contains the suggested label for the sample. If the sample is not a label issue (``issue == False``), the suggested label will be empty. For samples with label issues, the suggested label is computed by Cleanlab studio.

Outliers
========

Cleanlab Outlier
-------
The ``cleanlab_outlier`` column contains a boolean value, with ``True`` indicating that the sample is likely to be an outlier.