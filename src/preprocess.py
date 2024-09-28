import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    # Create a copy of the DataFrame to avoid modifying the original
    df = data.copy()

    known_cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G']
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) and x[0] in known_cabins else 'Unknown')

    # Fill missing values
    df["Embarked"] = df["Embarked"].fillna("S")  # Replace missing values with 'S'
    df["Fare"] = df["Fare"].replace(0, np.nan)  # Replace 0 with NaN in 'Fare'
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # Fill NaNs with the median
    df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill NaNs with the median

    # Extract titles from names
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Replace rare titles and standardize titles
    title_replacements = {
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Master": "Mr",
    }

    # Avoid chained assignment by assigning back to the DataFrame
    df["Title"] = df["Title"].replace(title_replacements)

    # Create a column for family size
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"] == 0).astype(bool)

    # Convert categorical variables to numerical variables
    df = pd.get_dummies(df, columns=["Sex", "Title", "Embarked", "Cabin"], drop_first=True)
    if 'Cabin_T' not in df.columns:
        cabin_t = pd.Series([False] * len(df), name='Cabin_T')
        position = len(df.columns) - 1
        df.insert(position, 'Cabin_T', cabin_t)

    # Normalize 'Age' and 'Fare' columns
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

    # Drop unnecessary columns
    df.drop(columns=["Ticket", "PassengerId", "Name"], inplace=True)

    return df
