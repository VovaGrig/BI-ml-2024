import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from IPython.display import display

NEW_LINE = "\n"


def run_eda(df: pd.DataFrame):
    """
    This function performs exploratory data analysis (EDA) on the input dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be analyzed.

    Returns:
    - None

    Prints the following information:
    - Number of observations and parameters in the dataframe
    - Categorical, numerical, and string variables in the dataframe
    - Statistics for categorical variables, including counts and frequencies
    - Statistics for numerical variables, including descriptive statistics, median, and IQR
    - Number of outliers for each numerical variable
    - Number of missing values in the dataframe
    - Number of duplicated rows in the dataframe
    """
    columns_types = [[], [], []]  # 0 - categorical, 1 - numeric, 2 - string
    print("Greetings, stranger! Please get acquainted with data\n")
    print(f"Number of observations: {df.shape[0]}")
    print(f"Number of parameters: {df.shape[1]}\n")
    for column in df.columns:
        if df[column].nunique() <= 12:
            columns_types[0].append(column)
        elif is_numeric_dtype(df[column]):
            columns_types[1].append(column)
        elif is_string_dtype(df[column]):
            columns_types[2].append(column)
        elif column == "Cabin":
            columns_types[2].append(column)
    print(f'Categorical variables: {", ".join(columns_types[0])}')
    print(f'Numerical variables: {", ".join(columns_types[1])}')
    print(f'String variables: {", ".join(columns_types[2])}{NEW_LINE}')
    print("Categorical variables statistics:")
    for categorical_variable in columns_types[0]:
        categorical_stats = pd.DataFrame()
        categorical_stats["counts"] = df[categorical_variable].value_counts()
        categorical_stats["frequencies"] = df[categorical_variable].value_counts(
            normalize=True
        )
        display(categorical_stats)
        print("\n")
    print("Numerical variables statistics:")
    numerical_stats = df[columns_types[1]].describe().T
    numerical_stats["median"] = df[columns_types[1]].median()
    numerical_stats["IQR"] = numerical_stats["75%"] - numerical_stats["25%"]
    display(numerical_stats)
    outliers = {}
    for numeric_variable in numerical_stats.index:
        outliers_number = df[
            (
                df[numeric_variable]
                <= numerical_stats.loc[numeric_variable, "25%"]
                - 1.5 * numerical_stats.loc[numeric_variable, "IQR"]
            )
            | (
                df[numeric_variable]
                >= numerical_stats.loc[numeric_variable, "75%"]
                + 1.5 * numerical_stats.loc[numeric_variable, "IQR"]
            )
        ].shape[0]
        if outliers_number > 0:
            outliers[numeric_variable] = outliers_number
    outliers = pd.DataFrame.from_dict(
        outliers, orient="index", columns=["Number of outliers"]
    )
    print("\n")
    display(outliers)
    print(f"Number of missing values: {df.isnull().sum().sum()}")
    print(f"Number of duplicated rows: {df.duplicated(keep=False).sum()}")
    return None
