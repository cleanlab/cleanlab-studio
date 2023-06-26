.. _concepts_cl_columns:

Cleanlab Columns
****************

Introduction
============

Cleanlab Studio will run a number of analyses on your dataset to help identify potential data issues. Currently we look for the following issues in datasets:
- Label issues
- Outliers
New analyses are being continuously added to Cleanlab Studio. The raw outputs of each analysis will be available to download as Cleanlab columns by calling ``studio.download_cleanlab_columns()``.

Most analyses produce two columns, one being a numeric score between 0 and 1 indicating the quality of a sample for that analysis (lower score means lower quality), and the other being a boolean, which indicates whether the sample is likely to have an issue or not. The different analyses are run separately, and hence they should be treated independently. For example, samples with label issues are not necessarily outliers, and vice versa, and it is possible to have a sample with label issue marked as ``False``, but outlier marked as ``True``. Furthermore, it is possible to have samples with label issue marked as ``True`` **and** outlier marked as ``True``: these are the raw results from the analyses, and you may interpret them as appropriate (e.g., if a data point is an outlier, it may be irrelevant that it's also likely a label issue).

The Cleanlab columns are listed below in the table. Detailed explanations of each column are provided in the following sections.

.. list-table:: 
   :widths: 50 25
   :header-rows: 1

   * - Cleanlab Column Name
     - Value Type
   * - :ref:`label_issue_score <label_issue_score>`
     - Float
   * - :ref:`is_label_issue <is_label_issue>`
     - Boolean
   * - :ref:`suggested_label <suggested_label>`
     - String
   * - :ref:`is_outlier <is_outlier>`
     - Boolean

Label Issues
============

.. _label_issue_score:
``label_issue_score``
-------------
Contains a score bounded between 0 and 1. The score is calculated using confident learning. The lower the score of a sample, the more likely it has a label issue.

.. _is_label_issue:
``is_label_issue``
-----
Contains a boolean value, with ``True`` indicating that the sample is likely to have a label issue. The value is obtained by thresholding the ``label_issue_score`` score with confident learning.

.. _suggested_label:
``suggested_label``
---------------
Contains the suggested label for the sample. If the sample is not a label issue (``is_label_issue == False``), the suggested label will be empty. For samples with label issues, the suggested label is computed by Cleanlab studio.

Outliers
========

.. _is_outlier:
``is_outlier``
-------
Contains a boolean value, with ``True`` indicating that the sample is likely to be an outlier.
Note: for tabular-mode multi-label projects, is_outlier will always be False. 