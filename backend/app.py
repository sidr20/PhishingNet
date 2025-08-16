import pandas as pd
import spacy
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

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

sms_df = pd.read_csv(r'C:\dev\PhishingNet\backend\SMSSpamCollection', sep='\t', names=['label', 'message'])
email_df = pd.read_csv(r'C:\dev\PhishingNet\backend\spam.csv')
email_df = email_df.rename(columns={'Category': 'label', 'Message': 'message'})
email_df['label'] = email_df['label'].map({'spam': 'spam', 'ham': 'ham'})
df = pd.concat([sms_df, email_df], ignore_index=True)
df = df.dropna()
df['features'] = df['message'].apply(extract_features)
X = np.stack(df['features'].values)
y = df['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = SVC(kernel='linear', probability=True)
model.fit(X_scaled, y)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        features = extract_features(message).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
