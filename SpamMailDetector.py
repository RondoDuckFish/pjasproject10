# Import necessary libraries

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk

# Download NLTK stopwords and tokenizer if not already available
nltk.download('stopwords')
nltk.download('punkt')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# stepo numero uno: Load the dataset
# Ensure the file name matches the dataset file in your project folder
data = pd.read_csv("email.csv", encoding='latin-1')  # This line should be at the start
data = data.rename(columns={'v1': 'label', 'v2': 'text'})  # Rename columns if needed
data = data[['label', 'text']]  # Select relevant columns
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)

# 2: Clean the text data
data['text'] = data['text'].apply(clean_text)

# 3: Split the dataset into training and testing sets
X = data['text']  # Features (email content)
y = data['label']  # Labels (spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4: Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)  # Use top 3000 words
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5: Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6: Test the model and evaluate performance
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy:", model.score(X_test_vec, y_test))

#7: Save the trained model
import joblib
joblib.dump(model, 'spam_model.pkl')

#8: Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

#9: Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#10: Use the loaded model and vectorizer to make predictions
new_email = "Hello, this is a spam email."
new_email_vec = vectorizer.transform([new_email])
prediction = model.predict(new_email_vec)
print("Prediction:", prediction)

#11: Use the loaded model and vectorizer to make predictions
new_email = "Hello, this is a ham email."
new_email_vec = vectorizer.transform([new_email])
prediction = model.predict(new_email_vec)
print("Prediction:", prediction)

def print_ascii_title():
    title = "SPAM EMAIL DETECTOR"
    ascii_art = r"""
     ____  ____    _    __  __    _    ____ _     ____  
    / ___||  _ \  / \  |  \/  |  / \  / ___| |   |  _ \ 
    \___ \| | | |/ _ \ | |\/| | / _ \| |   | |   | | | |
     ___) | |_| / ___ \| |  | |/ ___ \ |___| |___| |_| |
    |____/|____/_/   \_\_|  |_/_/   \_\____|_____|____/ 
    """
    print(ascii_art)

# Example usage
print_ascii_title()


# Display predictions for new emails
def display_results(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)
    result = "Spam" if prediction[0] == 1 else "Ham"
    print(f"Email: {email_text}\nPrediction: {result}\n")

# Example usage
display_results("Hello, this is a spam email.")
display_results("Hello, this is a ham email.")
def check_email_spam():
    email_text = input("Enter an email text to check if it's spam or not: ")
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)
    result = "Spam" if prediction[0] == 1 else "Ham"
    print(f"Email: {email_text}\nPrediction: {result}\n")

# Example usage
check_email_spam()

