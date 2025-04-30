#Import the required Libraries
import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
df = pd.read_csv(r'C:\Users\Lenovo\Email_Model\archive (5)\spam_assassin.csv') # Ensure this is the correct filename

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
print(df.head())

# Dataset Shape: (5796, 2)
# Dataset Columns: Index(['text', 'target'], dtype='object')
#                                                 text  target
# 0  From ilug-admin@linux.ie Mon Jul 29 11:28:02 2...       0
# 1  From gort44@excite.com Mon Jun 24 17:54:21 200...       1
# 2  From fork-admin@xent.com Mon Jul 29 11:39:57 2...       1
# 3  From dcm123@btamail.net.cn Mon Jun 24 17:49:23...       1
# 4  From ilug-admin@linux.ie Mon Aug 19 11:02:47 2...       0

# Check for missing values
df.dropna(inplace=True)

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply text cleaning
df["text"] = df["text"].apply(clean_text)

# Splitting data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features to avoid overfitting
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = nb_model.predict(X_test_tfidf)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#Accuracy: 0.9888

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.98      1.00      0.99       779
#            1       1.00      0.97      0.98       381

#     accuracy                           0.99      1160
#    macro avg       0.99      0.98      0.99      1160
# weighted avg       0.99      0.99      0.99      1160

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes Spam Detection")
plt.show()


# Testing with a new sample email
def predict_email(text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = nb_model.predict(text_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example test cases
email1 = "Congratulations! You won a lottery of $5000. Click the link to claim now."
email2 = "Hello, could you send me the meeting notes from our last discussion?"

print("Email 1 Prediction:", predict_email(email1))
print("Email 2 Prediction:", predict_email(email2))

# Email 1 Prediction: Spam
# Email 2 Prediction: Not Spam

#For Saving the model
import pickle

# Save the model
with open("spam_classifier.pkl", "wb") as model_file:
    pickle.dump(nb_model, model_file)

# Save the vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
