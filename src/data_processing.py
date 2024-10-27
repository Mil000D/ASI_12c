import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    df = pd.get_dummies(df, drop_first=True)
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
