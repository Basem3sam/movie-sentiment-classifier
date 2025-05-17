# 🎬 IMDB Sentiment Classifier

A machine learning pipeline that classifies movie reviews from the IMDB dataset as **positive** or **negative**.  
It uses text preprocessing with **NLTK**, feature extraction using **TF-IDF**, and a **Logistic Regression** classifier.

---

## 📚 Dataset

- **Source**: [Kaggle IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews (balanced: 25k positive, 25k negative)
- **Structure**:
  - `review`: text of the movie review
  - `sentiment`: either `positive` or `negative`

---

## 🧠 Project Pipeline

1. **Text Cleaning**  
   - Remove HTML tags, punctuation, and whitespace
   - Convert to lowercase

2. **Preprocessing** (NLTK)  
   - Remove stopwords
   - Lemmatize words using WordNet

3. **Feature Extraction**  
   - Use `TfidfVectorizer` to transform text to feature vectors (top 5000 words)

4. **Model Training**  
   - Binary classification using `LogisticRegression` from scikit-learn

5. **Evaluation**  
   - Classification Report
   - Confusion Matrix

---

## 📈 Results

- **Accuracy**: ~90%
- **Evaluation Tools**:
  - `classification_report`: precision, recall, f1-score
  - `confusion_matrix` visualization with `ConfusionMatrixDisplay`

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Basem3sam/movie-sentiment-classifier.git
cd movie-sentiment-classifier
```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
3. Download NLTK data (only once):
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4.Run the training script:
```bash
python model/train.py
```
Or open the interactive notebook:
```bash
jupyter notebook notebook/exploration.ipynb
```

---

## 📂 Project Structure
```bash
movie-sentiment-classifier/
│
├── data/                  # (excluded in .gitignore)
│   └── IMDB Dataset.csv
├── model/
│   └── train.py           # Python script for training
├── notebook/
│   └── exploration.ipynb  # Full analysis notebook
├── assets/                # Plots and images (optional)
├── requirements.txt
├── .gitignore
└── README.md              # Project documentation
```

---

## ⚙️ Requirements
Python 3.8+

pandas

scikit-learn

nltk

matplotlib

Install all with:
```bash
pip install -r requirements.txt
```

---

## ✍️ Author
LinkedIn: [BasemEsam](linkedin.com/in/basemesam)
GitHub: [Basem3sam](github.com/basem3sam)

---

## 📌 Notes
Dataset should be placed in data/IMDB Dataset.csv

The project uses relative paths (../data/...) – keep the directory structure intact

NLTK stopwords and WordNet need to be downloaded once

---

## ✅ Summary
✔️ Cleaned & preprocessed IMDB reviews
✔️ Extracted TF-IDF features
✔️ Trained a logistic regression model
✔️ Achieved ~90% accuracy
✔️ Ready for deployment or further optimization 🚀
