import pandas as pd
from src.config import LABEL_FILE


def read_labels():
    """
    Read the dataset label file.

    Returns:
        pandas.DataFrame: A dataframe containing video names and real speeds.
    """

    # Read ODS label file
    labels_df = pd.read_excel(LABEL_FILE, engine="odf")

    # Rename columns to simpler English names
    labels_df = labels_df.rename(
        columns={
            "Number": "number",
            "Name": "dataset_name",
            "Real (m/s)": "real_speed"
        }
    )

    # Convert speed column to numeric values
    labels_df["real_speed"] = pd.to_numeric(labels_df["real_speed"], errors="coerce")

    return labels_df
