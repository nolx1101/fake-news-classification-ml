import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model1.joblib')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer1.joblib')

# Load the test data
test_df = pd.read_csv('test.csv')
test_df.fillna('', inplace=True)  # Fill missing values

# Preprocess and vectorize the test data
tfidf_test = tfidf_vectorizer.transform(test_df['text'])

# Make predictions
predictions = model.predict(tfidf_test)

# Save predictions to fake_news_predictions.csv
submission_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
submission_df.to_csv('fake_news_predictions.csv', index=False)

# Confirmation message
print("Submission file 'fake_news_predictions.csv' has been created successfully.")
