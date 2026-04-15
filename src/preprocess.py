import re
import pandas as pd


def clean_text(text):
    """
    Clean raw review text
    """

    text = text.lower()

    # remove HTML tags like <br />
    text = re.sub(r"<br\s*/?>", " ", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df, text_column="review"):
    """
    Apply cleaning to a dataframe column
    """

    df["clean_review"] = df[text_column].apply(clean_text)

    return df