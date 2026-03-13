import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset from a CSV file.
df = pd.read_csv('train.csv')

# Data Preprocessing
# 'text' contains the news articles, and 'label' indicates if the news is real (1) or fake (0).
X = df['text']
y = df['label']

# Splitting the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill any missing values in the text columns with empty strings to maintain data consistency.
X_train = X_train.fillna('')
X_test = X_test.fillna('')

# Feature Extraction: Transforming text data into TF-IDF features.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF vectorizer.
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer1.joblib')

# Model Selection and Training: Multinomial Naive Bayes.
model = MultinomialNB()
model.fit(tfidf_train, y_train)

# Save the trained model.
joblib.dump(model, 'model1.joblib')

# Predict and evaluate with trained model.
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Displaying the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
