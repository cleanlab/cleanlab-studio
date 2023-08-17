.. _concepts_cl_columns:

Cleanlab Columns
****************

Introduction
============

Cleanlab Studio will run a number of analyses on your dataset to help identify potential data issues. Currently we run the following analyses:

- Label issues
- Outliers
- Ambiguous examples
- well-labeled examples
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
| well-labeled Examples   | :ref:`_is_well_labeled <_is_well_labeled>` | Boolean     |
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

well-labeled
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


Project Information Columns
============


You can also retrieve information/metadata related to the Cleanlab project itself.

- Action Taken
- If a row previously had an issue
- Top Labels and Probabilities (alternative suggested labels and how likely they are)
- The originally assigned (given) label and how likely it is to be the true label
- The predicted label and its likelihood (even when we don't suggest changing from the given label).
- If the row was initially unlabeled/had no given label. 

To get this information, use ``studio.download_cleanlab_columns()`` and add the argument -p or -a for project details or all columns including analysis columns respectively.


+-------------------------------+------------------------------------------------------------------+
| Project Detail                | Cleanlab Column Name                              | Value Type   |
+===============================+==================================================================+
| Action                        | :ref:`action <action>`                            | String       |
+-------------------------------+------------------------------------------------------------------+
| Corrected Label               | :ref:`corrected_label <corrected_label>`          | String       |
+-------------------------------+------------------------------------------------------------------+
| Previously Resolved           | :ref:`previously_resolved <previously_resolved>`  | Boolean      |
+-------------------------------+------------------------------------------------------------------+
| Top Labels                    | :ref:`top_labels <top_labels>`                    | List[String] |
+-------------------------------+------------------------------------------------------------------+
| Top Probs                     | :ref:`top_probs <top_probs>`                      | List[Float]  |
+-------------------------------+------------------------------------------------------------------+
| Given Label                   | :ref:`given_label <given_label>`                  | String       |
+-------------------------------+------------------------------------------------------------------+
| Likelihood of Given Label     | :ref:`p_given_label <p_given_label>`              | Float        |
+-------------------------------+------------------------------------------------------------------+
| Predicted Label               | :ref:`predicted_label <predicted_label>`          | String       |
+-------------------------------+------------------------------------------------------------------+
| Likelihood of Predicted Label | :ref:`p_predicted_label <p_predicted_label>`      | Float        |
+-------------------------------+------------------------------------------------------------------+
| Initially Unlabeled           | :ref:`initially_unlabeled <initially_unlabeled>`  | Boolean      |
+-------------------------------+------------------------------------------------------------------+
Project Details
============

.. _action:
``action``
---------
Contains a description of the action taken on this row, which can be one of Unresolved (no action taken for a row where there is an issue), Exclude (remove row from dataset), Label (assign a label to the row), Keep (retain the given label),Auto-fix (take the action recommended by Cleanlab - can be one of [exclude, label, keep]), or None (no action taken on a row with no issues).
.. _corrected_label:
``corrected_label``
---------------
Contains the corrected label. If you use Cleanlab to resolve a label issue, either manually in the resolver panel or via auto-fix, this is where that label resides.
.. _previously_resolved:
``previously resolved``
-------------
Contains a boolean value which is 1 if an action was taken on this row in a previous Cleanset version (i.e., before using the Improve Results feature), and 0 otherwise. 
.. _top_labels:
``top_labels``
---------------
Contains a list of all labels with at least 1% likelihood as assessed by Cleanlab, sorted from most to least likely, including the suggested label if one exists.

.. _top_probs:
``top_probs``
---------------
Contains a list of the probabilities of each of label in top_labels being the true label. 

.. _given_label:
``given_label``
----------
Contains the originally assigned label on project creation. 

.. _p_given_label:
``p_given_label``
----------
Contains the likelihood of the given label being the true label as calculated by Cleanlab.

.. _predicted_label:
``predicted_label``
----------
Contains the highest likelihood label for this row. Note that this is not the same as the suggested label! In many cases, Cleanlab finds that a data point/row is unlikely to have an issue and therefore does suggest using the predicted label. This is done using confident learning, and ensures that Cleanlab defers to the given label when it is appropriate. Use the predicted_label only for evaluation purposes.

.. _p_predicted_label:
``p_predicted_label``
----------
Contains the likelihood of the predicted label being the true label as calculated by Cleanlab.


.. _initially_unlabeled:
``initially_unlabeled``
-------------
Contains a boolean indicating whether or not this row originally had a label associated with it. Rows which are initially unlabeled naturally do not have a given label, so some analyses (e.g., is_high_confidence_given_label) will not be available. You can use Cleanlab as a data labeling platform in this case!
