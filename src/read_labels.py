import pandas as pd

from src.config import LABEL_FILE


def read_labels():
    labels_df = pd.read_excel(
        LABEL_FILE,
        engine="odf",
    )

    labels_df = labels_df.rename(
        columns={
            "Number": "number",
            "Name": "dataset_name",
            "Real (m/s)": "real_speed",
        }
    )

    labels_df["real_speed"] = pd.to_numeric(
        labels_df["real_speed"],
        errors="coerce",
    )

    return labels_df
