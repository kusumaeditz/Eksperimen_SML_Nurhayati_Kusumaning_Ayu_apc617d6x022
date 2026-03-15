import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded: {df.shape}")
    return df

def handle_missing_values(df):
    df = df.copy()
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    print(f"[INFO] Missing values handled")
    return df

def remove_duplicates(df):
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Removed {before - len(df)} duplicates")
    return df

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    print(f"[INFO] Outliers removed. Shape: {df.shape}")
    return df

def encode_features(df):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['str']).columns.tolist()
    if not categorical_cols:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    print(f"[INFO] Encoded: {categorical_cols}")
    return df

def scale_features(df, target_col='loan_status'):
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"[INFO] Features scaled")
    return df

def split_and_save(df, output_dir, target_col='loan_status'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(output_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"[INFO] Saved to {output_dir}")
    print(f"[INFO] Train: {train_df.shape}, Test: {test_df.shape}")

def preprocess(input_path, output_dir):
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, ['person_age', 'person_income', 'person_emp_length'])
    df = encode_features(df)
    df = scale_features(df)
    split_and_save(df, output_dir)
    print("[INFO] Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../credit_risk_dataset.csv')
    parser.add_argument('--output', type=str, default='credit_risk_dataset_preprocessing')
    args = parser.parse_args()
    preprocess(args.input, args.output)