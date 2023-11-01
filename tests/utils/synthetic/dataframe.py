import pandas as pd
from io import StringIO


def load_cleanset_df() -> pd.DataFrame:
    """Loads the cleanset.csv file into a pandas DataFrame.

    Note:
        It contains the following columns:
        - id: The unique identifier of the example
        - real_or_synthetic: Whether the example is real or synthetic
        - label_issue_score: The label issue score of the example (either "real" or "synthetic")
        - near_duplicate_cluster_id: The ID of the near duplicate cluster that the example belongs to (if it's a near duplicate)
        - is_near_duplicate: Whether the example is a near duplicate or not
        - text: The text of the example

    Warning:
        This dataframe only contains the necessary cleanlab columns expected from Cleanlab Studio.
        This is just for testing purposes.
    """

    csv_data = """id,real_or_synthetic,label_issue_score,near_duplicate_cluster_id,is_near_duplicate,text
        1,real,0.8,,False,"This is a decent real example that needs more synthetic counterparts."
        2,real,0.6,,False,"This is another real example."
        3,real,0.9,,False,"This is a real example that's likely underrepresented in the synthetic dataset."
        4,real,0.7,2,False,"This is a real example that's memorized."
        5,real,0.5,,False,"This is a real example."
        6,real,0.4,,False,"This is another real example."
        7,real,0.3,,False,"A real example."
        8,real,0.2,,False,"This is a real example that isn't discriminated from the synthetic dataset."
        9,real,0.1,,False,"This is a real example that's not confidently distinguished from the synthetic dataset."
        10,real,0.0,,False,"This is a real example that's well represented in the synthetic dataset."
        11,synthetic,0.8,,False,"This is a decent synthetic example."
        12,synthetic,0.6,,False,"This is another synthetic example."
        13,synthetic,0.9,,False,"This is a great synthetic example. Looks real."
        14,synthetic,0.5,1,True,"This is a synthetic example. Unvaried."
        15,synthetic,0.7,1,True,"This is a SYNTHETIC example. Unvaried."
        16,synthetic,0.4,2,False,"This is a real example that's memorized. Unoriginal."
        17,synthetic,0.3,,False,"This is clearly a synthetic example."
        18,synthetic,0.2,,False,"This is obviously a synthetic example."
        19,synthetic,0.1,,False,"This is not a good synthetic example."
        20,synthetic,0.0,,False,"This is an unrealistic synthetic example."
        """

    return pd.read_csv(StringIO(csv_data))
