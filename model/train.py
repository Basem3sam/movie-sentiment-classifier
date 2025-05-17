import pandas as pd
import re
import nltk
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download NLTK data (only needs to run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Get the current file's directory (i.e., where train.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct a path to the CSV relative to this script
data_path = os.path.join(current_dir, '..', 'data', 'IMDB Dataset.csv')

# Load dataset
if not os.path.exists(data_path):
    print("Current working directory:", os.getcwd())
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = re.sub(r'<.*?>', '', text)              # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)         # Keep only letters
    text = text.lower()                            # Convert to lowercase
    tokens = text.split()                          # Split into words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Clean and preprocess the reviews
print("Cleaning and preprocessing text...")
df['review_clean'] = df['review'].apply(preprocess)

# Convert sentiment labels to binary
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['review_clean'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize text using TF-IDF
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict on test set
y_pred = model.predict(X_test_vec)

# Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved.")
