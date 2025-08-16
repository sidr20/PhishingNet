import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

nlp = spacy.load('en_core_web_md')


def extract_features(text):
    semantic_vector = nlp(text).vector
    custom_features = [
        len(text),
        len(re.findall(r'[!]', text)),
        len(re.findall(r'[$€£]', text)),
        len(re.findall(r'\d', text)) / (len(text) + 1e-5),
        len(re.findall(r'[A-Z]', text)) / (len(text) + 1e-5),
        1 if 'http' in text or 'www' in text  or 'scam' in text else 0
    ]
    return np.concatenate((semantic_vector, np.array(custom_features)))
sms_df = pd.read_csv('sms+spam+collection/SMSSpamCollection', sep='\t', names=['label', 'message'])
email_df = pd.read_csv('spam.csv')
email_df = email_df.rename(columns={'Category': 'label', 'Message': 'message'})
email_df['label'] = email_df['label'].map({'spam': 'spam', 'ham': 'ham'})
df = pd.concat([sms_df, email_df], ignore_index=True)
df = df.dropna()

df['features'] = df['message'].apply(extract_features)
X = np.stack(df['features'].values)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

print("\n--- Test a New Message ---")
user_message = input("Enter the email or text: ")

vectorized_message = extract_features(user_message).reshape(1, -1)
prediction = model.predict(vectorized_message)

print(f"This message is a '{prediction[0]}' message.")