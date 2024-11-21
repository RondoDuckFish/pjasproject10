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
