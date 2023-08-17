
.. _concepts_project_information:

Project Information Columns
****************

Introduction
============

For each project that you run in Cleanlab Studio, Cleanlab stores a number of metadata columns that can be retrieved using the Python API. 
Currently we store the following project information columns:

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
| Previously Resolved           | :ref:`is_previously_resolved <previously_resolved>`  | Boolean      |
+-------------------------------+------------------------------------------------------------------+
| Top Labels                    | :ref:`top_labels <top_labels>`                    | List[String] |
+-------------------------------+------------------------------------------------------------------+
| Top Probs                     | :ref:`top_probs <top_probs>`                      | List[Float]  |
+-------------------------------+------------------------------------------------------------------+
| Given Label                   | :ref:`given_label <given_label>`                  | String       |
+-------------------------------+------------------------------------------------------------------+
| Likelihood of Given Label     | :ref:`given_label_prob <given_label_prob>`              | Float        |
+-------------------------------+------------------------------------------------------------------+
| Predicted Label               | :ref:`predicted_label <predicted_label>`          | String       |
+-------------------------------+------------------------------------------------------------------+
| Likelihood of Predicted Label | :ref:`predicted_label_prob <predicted_label_prob>`      | Float        |
+-------------------------------+------------------------------------------------------------------+
| Initially Unlabeled           | :ref:`is_initially_unlabeled <is_initially_unlabeled>`  | Boolean      |
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
.. _is_previously_resolved:
``is_previously resolved``
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

.. _given_label_prob:
``p_given_label``
----------
Contains the likelihood of the given label being the true label as calculated by Cleanlab.

.. _predicted_label:
``predicted_label``
----------
Contains the highest likelihood label for this row. Note that this is not the same as the suggested label! In many cases, Cleanlab finds that a data point/row is unlikely to have an issue and therefore does suggest using the predicted label. This is done using confident learning, and ensures that Cleanlab defers to the given label when it is appropriate. Use the predicted_label only for evaluation purposes.

.. _predicted_label_prob:
``p_predicted_label``
----------
Contains the likelihood of the predicted label being the true label as calculated by Cleanlab.


.. _is_initially_unlabeled:
``is_initially_unlabeled``
-------------
Contains a boolean indicating whether or not this row originally had a label associated with it. Rows which are initially unlabeled naturally do not have a given label, so some analyses (e.g., is_high_confidence_given_label) will not be available. You can use Cleanlab as a data labeling platform in this case!
