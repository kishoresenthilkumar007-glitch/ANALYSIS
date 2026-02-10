import sys 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()
    df['avg_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
    df['pass'] = (df['avg_score'] >= 60).astype(int)
    # One-hot encode categorical columns
    cat_cols = ['gender', 'parental_education', 'lunch', 'test_prep_course']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def eda(df, outdir):
    print('Rows, cols:', df.shape)
    print('\nHead:\n', df.head())
    print('\nDescribe:\n', df[['math_score','reading_score','writing_score','avg_score']].describe())

    plt.figure(figsize=(8,6))
    sns.histplot(df['avg_score'], kde=True)
    plt.title('Average Score Distribution')
    plt.savefig(os.path.join(outdir, 'avg_score_hist.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(df[['math_score','reading_score','writing_score','avg_score']].corr(), annot=True, cmap='coolwarm')
    plt.title('Score Correlation')
    plt.savefig(os.path.join(outdir, 'score_correlation.png'))
    plt.close()


def train_and_evaluate(df):
    features = df.drop(columns=['avg_score','pass'])
    target = df['pass']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f'Accuracy: {acc:.3f}')
    print('\nClassification Report:\n', classification_report(y_test, preds, digits=3))
    cm = confusion_matrix(y_test, preds)
    print('\nConfusion Matrix:\n', cm)
    return clf, X_test, y_test, preds


def save_confusion_matrix(cm, outpath):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(outpath)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print('Usage: python analysis.py data/sample_students.csv')
        sys.exit(1)
    data_path = sys.argv[1]
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok=True)

    df = load_data(data_path)
    df_proc = preprocess(df)
    eda(df_proc, outdir)
    clf, X_test, y_test, preds = train_and_evaluate(df_proc)
    cm = confusion_matrix(y_test, preds)
    save_confusion_matrix(cm, os.path.join(outdir, 'confusion_matrix.png'))
    print(f'Plots and outputs saved to {outdir}')


if __name__ == '__main__':
    main()
