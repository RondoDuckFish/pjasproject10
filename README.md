v1.1 --> added these following lines of code
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

- v1.1overall makes all of the stuff of the model run better

v1.0 --> made actual code for training naive bayes model using bayesian statistics 

 **pjasproject10**
pjas project on ml algorithm on spam mail detection
21-11-2024 -> I am attemtping to make a spam mail detection system/algorithm for a PJAS science project so far I have built the base python program and given users detail on what they should be doing at the terminal, so people of all experience levels can do this stuff and sort of understand how it works ( ͡~ ͜ʖ ͡°)
