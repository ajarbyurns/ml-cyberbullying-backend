import os
import re
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
ENCODER_DIR = os.path.join(PROJECT_ROOT, 'encoders')
MODEL_NAME_PATTERN = r'text_classifier_v(\d+)\.pkl'
ENCODER_NAME_PATTERN = r'label_encoder_v(\d+)\.pkl'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cyberbullying_tweets.csv')

def get_last_model_version(model_dir=MODEL_DIR) -> int:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    files = os.listdir(model_dir)
    versions = [int(match.group(1))
                for f in files
                if (match := re.search(MODEL_NAME_PATTERN, f))]
    return max(versions, default=1)

def get_last_encoder_version(encoder_dir=ENCODER_DIR) -> int:
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)

    files = os.listdir(encoder_dir)
    versions = [int(match.group(1))
                for f in files
                if (match := re.search(ENCODER_NAME_PATTERN, f))]
    return max(versions, default=1)

def train_model(data_path=DATA_PATH, new_model=False, new_encoder=False):
    print("Loading data...")
    df = pd.read_csv(data_path)

    if 'tweet_text' not in df.columns or 'cyberbullying_type' not in df.columns:
        raise ValueError("Dataset must contain 'tweet_text' and 'cyberbullying_type' columns.")

    X = df['tweet_text']
    y = df['cyberbullying_type']

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("Building pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, average='micro')
    prec = precision_score(y_test, preds, average='micro')
    print(f"Model accuracy: {acc:.4f}")
    print(f"Model precision: {prec:.4f}")
    print(f"Model recall: {rec:.4f}")

    model_version = get_last_model_version()
    if new_model:
        model_version += 1
    model_path = os.path.join(MODEL_DIR, f'text_classifier_v{model_version}.pkl')

    encoder_version = get_last_encoder_version()
    if new_encoder:
        encoder_version += 1
    label_path = os.path.join(ENCODER_DIR, f'label_encoder_v{encoder_version}.pkl')

    print("Saving model and label encoder...")
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, label_path)

    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {label_path}")

if __name__ == "__main__":
    train_model()
