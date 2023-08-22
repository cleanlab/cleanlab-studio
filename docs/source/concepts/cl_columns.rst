.. _concepts_cl_columns:

Cleanlab Columns
****************

Introduction
============

Cleanlab Studio will run a number of analyses on your dataset to help identify potential data issues. Currently we run the following analyses:

- Label issues
- Outliers
- Ambiguous examples
- Well labeled examples
- Near duplicates (only for image and text datasets)

New analyses are being continuously added to Cleanlab Studio. The raw outputs of each analysis will be available to download as Cleanlab columns by calling ``studio.download_cleanlab_columns()``.

Most analyses produce two columns, one being a numeric score between 0 and 1 indicating the quality of a sample for that analysis (lower score means lower quality), and the other being a boolean, which indicates whether the sample is likely to have an issue or not. When both the score and boolean are present, the boolean is thresholded from the corresponding score, with the threshold chosen intelligently based on benchmarks on a large number of datasets.

The different analyses are run separately, and hence they should be treated independently. For example, samples with label issues are not necessarily outliers, and vice versa, and it is possible to have a sample with label issue marked as ``False``, but outlier marked as ``True``. Furthermore, it is possible to have samples with label issue marked as ``True`` **and** outlier marked as ``True``: these are the raw results from the analyses, and you may interpret them as appropriate (e.g., if a data point is an outlier, it may be irrelevant that it's also likely a label issue).

The Cleanlab columns are listed below in the table. Detailed explanations of each column are provided in the following sections.


+----------------------------+------------------------------------------------------------------------+-------------+
| Analysis Type              | Cleanlab Column Name                                                   | Value Type  |
+============================+========================================================================+=============+
| Label Issues               | :ref:`is_label_issue <is_label_issue>`                                 | Boolean     |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`label_issue_score <label_issue_score>`                           | Float       |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`suggested_label <suggested_label>`                               | String      |
+----------------------------+------------------------------------------------------------------------+-------------+
| Ambiguous Examples         | :ref:`is_ambiguous <is_ambiguous>`                                     | Boolean     |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`ambiguous_score <ambiguous_score>`                               | Float       |
+----------------------------+------------------------------------------------------------------------+-------------+
| Well labeled Examples      | :ref:`is_well_labeled <_is_well_labeled>`                              | Boolean     |
+----------------------------+------------------------------------------------------------------------+-------------+
| Near Duplicates            | :ref:`is_near_duplicate <is_near_duplicate>`                           | Boolean     |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`near_duplicate_score <near_duplicate_score>`                     | Float       |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`near_duplicate_id <near_duplicate_id>`                           | Integer     |
+----------------------------+------------------------------------------------------------------------+-------------+
| Outliers                   | :ref:`is_outlier <is_outlier>`                                         | Boolean     |
+                            +------------------------------------------------------------------------+-------------+
|                            | :ref:`outlier_score <outlier_score>`                                   | Float       |
+----------------------------+------------------------------------------------------------------------+-------------+


Label Issues
============

.. _is_label_issue:
``is_label_issue``
-----
Contains a boolean value, with ``True`` indicating that the sample is likely to have a label issue. The value is obtained by thresholding the ``label_issue_score`` score with confident learning.

.. _label_issue_score:
``label_issue_score``
-------------
Contains a score bounded between 0 and 1. The score is calculated using confident learning. The lower the score of a sample, the more likely it has a label issue.

.. _suggested_label:
``suggested_label``
---------------
Contains the suggested label for the sample. If the sample is not a label issue (``is_label_issue`` marked as ``False``), the suggested label will be empty. For samples with label issues, the suggested label is computed by Cleanlab studio.


Ambiguous
=========

.. _is_ambiguous:
``is_ambiguous``
----------
Contains a boolean value, with ``True`` indicating that the sample is likely to be ambiguous. Ambiguous samples are those that do not obviously belong to a single class.

.. _ambiguous_score:
``ambiguous_score``
-------------
Contains a score bounded between 0 and 1, which is used to determine whether a sample is ambiguous. The lower the score of a sample, the more likely it is to be ambiguous.

Well labeled
===============

.. _is_well_labeled:
``is_well_labeled``
---------------
Contains a boolean value, with ``True`` indicating that the given label of the sample is highly likely to be correct, so the sample can be safely used in downstream tasks.

Near Duplicates
===============
*Note: Near-duplicates are not computed for tabular-type datasets.*

.. _is_near_duplicate:
``is_near_duplicate``
----------------
Contains a boolean value, with ``True`` indicating that the sample is likely to be a near duplicate of another sample. Near duplicates are two or more examples in a dataset that are extremely similar (or identical) to each other, relative to the rest of the dataset.

.. _near_duplicate_score:
``near_duplicate_score``
------------------
Contains a score bounded between 0 and 1, which is used to determine whether a sample is a near duplicate. The lower the score of a sample, the more likely it is to be a near duplicate of another sample.

.. _near_duplicate_id:
``near_duplicate_id``
----------------
Contains an integer ID for each sample, where samples with the same ID are near duplicates of each other. The IDs range from 0 upwards. Samples that do not have near duplicates are assigned an ID of `<NA>`.


Outliers
========
*Note: for projects on multi-label tabular datasets, outliers are currently not computed.*

.. _is_outlier:
``is_outlier``
-------
Contains a boolean value, with ``True`` indicating that the sample is likely to be an outlier.

.. _outlier_score:
``outlier_score``
-----------
Contains a score bounded between 0 and 1, which is used to determine whether a sample is an outlier. The lower the score of a sample, the more likely it is to be an outlier.
