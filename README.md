# Twitter US Airline Sentiment Analysis

<div align="center">

![Project Banner](/images/Sentiment_Analysis.jpg)

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Predicting sentiment (negative, neutral, positive) from airline tweets using MultinomialNB, LinearSVC, and RandomForest**

[Key Findings](#-key-findings) • [Dataset](#-dataset) • [Notebooks](#-notebooks) • [Results](#-model-performance) • [Usage](#-installation--usage)

</div>

---

## About The Project

Customer sentiment on social media is a critical indicator of service quality for airlines. Understanding how passengers feel about their flying experience enables companies to identify pain points and improve customer satisfaction.

This project uses **supervised text classification** to predict sentiment (negative, neutral, positive) from 14,640 tweets about 6 major US airlines. Three models are systematically compared: **MultinomialNB**, **LinearSVC**, and **RandomForest**, with hyperparameter tuning via GridSearchCV.

The key finding is that **LinearSVC achieves the best performance (80.36% accuracy)**, demonstrating that linear models with TF-IDF features are highly effective for sentiment classification on social media text.

### Objectives

1. Explore patterns and characteristics of airline tweets
2. Compare multiple classification models on text data
3. Identify the best model for deployment with error analysis

---

## Key Findings

### Can we predict tweet sentiment?

**YES!** LinearSVC achieves strong performance:

| Metric | LinearSVC | MultinomialNB | RandomForest |
|--------|:---------:|:-------------:|:------------:|
| **Test Accuracy** | **0.8036** | 0.6916 | 0.7531 |
| **F1-Score (macro)** | **0.74** | 0.62 | 0.68 |
| **CV Mean** | **0.7289** | 0.6618 | 0.6533 |

- **Accuracy = 80.36%** — correctly predicts sentiment for 4 out of 5 tweets
- **F1-Score macro = 0.74** — balanced performance across all 3 classes
- **Negative sentiment** is easiest to detect (F1 = 0.88)
- **Neutral sentiment** is hardest to detect (F1 = 0.62) due to limited data

### What drives predictions?

Top factors influencing sentiment prediction:

1. **Negative words** — "worst", "lost", "cancelled", "delayed" strongly indicate negative sentiment
2. **Positive words** — "amazing", "best", "thank", "love" indicate positive sentiment
3. **Airline mentions** — @united, @americanair, @usairways appear more in negative tweets
4. **Text length** — Negative tweets tend to be longer (more detailed complaints)

> **Business Insight**: Customer Service Issues and Late Flight are the top reasons for negative sentiment. Airlines should prioritize these areas for immediate improvement.

---

## Dataset

**Source**: [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) by Crowdflower (originally from Twitter)

**Description**: Tweets about 6 US airlines, manually classified by crowdworkers into positive, negative, or neutral sentiment.

**Statistics**:

| Property | Value |
|----------|-------|
| Samples | 14,640 tweets |
| Features | 15 columns |
| Target | `airline_sentiment` (negative, neutral, positive) |
| Text Column | `text` (raw tweet text) |
| Airlines | United, US Airways, American, Southwest, Delta, Virgin America |

**Class Distribution**:

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Negative | 9,178 | 62.7% |
| Neutral | 3,099 | 21.2% |
| Positive | 2,363 | 16.1% |

→ **Imbalanced dataset** — negative tweets dominate, so F1-score macro is more appropriate than accuracy.

**Features**:

| # | Feature | Description |
|---|---------|-------------|
| 1 | tweet_id | Unique tweet identifier |
| 2 | airline_sentiment | Target: negative, neutral, positive |
| 3 | airline_sentiment_confidence | Confidence of sentiment label |
| 4 | airline | Airline name (6 airlines) |
| 5 | name | Twitter username |
| 6 | text | Raw tweet text |
| 7 | retweet_count | Number of retweets |
| 8 | tweet_created | Tweet timestamp |
| 9 | tweet_location | User's location |
| 10 | user_timezone | User's timezone |

---

## Project Structure

```
Twitter US Airline Sentiment/
├── data/
│   ├── raw/                              # Raw dataset
│   │   └── Tweets.csv                    # Source data from Kaggle
│   └── processed/                        # Processed data
│       ├── cleaned_data.csv              # Cleaned text data
│       ├── X_tfidf.npz                   # TF-IDF sparse matrix
│       └── y_labels.npy                  # Encoded labels
├── notebooks/                            # Jupyter notebooks (numbered)
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb            # Text Preprocessing
│   ├── 03_feature_extraction.ipynb       # TF-IDF Feature Extraction
│   └── 04_modeling.ipynb                 # Modeling & Evaluation
├── models/                               # Trained models
│   ├── model_final.pkl                   # Best model (LinearSVC)
│   ├── tfidf_vectorizer.pkl              # TF-IDF vectorizer
│   └── label_encoder.pkl                 # Label encoder
├── images/                               # Exported visualizations
│   ├── sentiment_distribution.png
│   ├── wordcloud_negative.png
│   ├── model_comparison.png
│   ├── confusion_matrix_best.png
│   └── confusion_matrix_all.png
├── requirements.txt                      # Dependencies
├── README.md                             # Project documentation
└── .gitignore
```

---

## Notebooks

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Sentiment distribution analysis
- Tweet length analysis by sentiment
- Top reasons for negative tweets
- Airline comparison
- **Key Insights**:
  - 62.7% of tweets are negative — most passengers tweet when unhappy
  - Negative tweets are longer (avg 114 chars) vs positive (avg 86 chars)
  - Customer Service Issue is the #1 reason for negative tweets (2,910 occurrences)

### 2. Text Preprocessing (`02_preprocessing.ipynb`)
- **Lowercasing** — normalize text to lowercase
- **URL removal** — remove http/https links
- **Mention removal** — remove @username
- **Hashtag removal** — remove #hashtags
- **Punctuation removal** — remove special characters
- **Number removal** — remove digits
- **Whitespace normalization** — remove extra spaces
- **Text length reduction**: 21.2% average reduction (103.8 → 81.8 chars)

### 3. Feature Extraction (`03_feature_extraction.ipynb`)
- **TF-IDF Vectorization** with optimized parameters
- **Parameters**: ngram_range=(1,2), max_features=50000, sublinear_tf=True
- **Output**: Sparse matrix (14,640 × 27,780 features)
- **Label Encoding**: negative=0, neutral=1, positive=2

### 4. Modeling & Evaluation (`04_modeling.ipynb`)
- **7 models compared**: 3 baseline + 3 tuned + 1 ensemble
- **Hyperparameter tuning** with GridSearchCV (5-fold CV)
- **Confusion matrix** analysis for all models
- **Error analysis** with example misclassifications
- **Best model**: LinearSVC (Accuracy: 80.36%)

---

## Model Performance

### Overall Comparison

| Rank | Model | Accuracy | F1-Score (macro) | CV Mean |
|------|-------|:--------:|:-----------------:|:-------:|
| 1 | **LinearSVC** | **0.8036** | **0.74** | **0.7289** |
| 2 | SVC (tuned) | 0.7992 | 0.73 | 0.7289 |
| 3 | NB (tuned) | 0.7698 | 0.68 | 0.6618 |
| 4 | RF (tuned) | 0.7589 | 0.69 | 0.6533 |
| 5 | Ensemble (Voting) | 0.7538 | 0.67 | — |
| 6 | RandomForest | 0.7531 | 0.68 | 0.6533 |
| 7 | MultinomialNB | 0.6916 | 0.62 | 0.6618 |

### Best Model: LinearSVC

**Hyperparameters**:
```python
{'C': 1.0, 'max_iter': 10000, 'random_state': 43}
```

**Performance Highlights**:
- Accuracy = 80.36% — correctly predicts 4 out of 5 tweets
- Negative sentiment F1 = 0.88 — excellent detection
- Neutral sentiment F1 = 0.62 — hardest class due to limited data
- Training time: < 1 second — very fast

**Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Negative | 0.84 | 0.91 | 0.88 | 1,835 |
| Neutral | 0.66 | 0.58 | 0.62 | 620 |
| Positive | 0.78 | 0.65 | 0.71 | 473 |
| **Accuracy** | | | **0.80** | **2,928** |

**Why LinearSVC Wins**:
- Linear kernel is ideal for high-dimensional sparse text data (TF-IDF)
- Faster training than non-linear SVM alternatives
- Better generalization than tree-based models on text data
- Simple and interpretable — suitable for production deployment

---

## Installation & Usage

### Prerequisites
- Python 3.13+
- Jupyter Notebook

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Twitter-US-Airline-Sentiment
cd Twitter-US-Airline-Sentiment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in sequence:
1. `01_eda.ipynb` — Data exploration
2. `02_preprocessing.ipynb` — Text cleaning
3. `03_feature_extraction.ipynb` — TF-IDF features
4. `04_modeling.ipynb` — Model training & evaluation

### Load Trained Model

```python
import joblib
import re
import string

# Load model and tools
model = joblib.load('models/model_final.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
le = joblib.load('models/label_encoder.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Predict
tweet = "@united worst flight ever! Lost my luggage."
clean = preprocess_text(tweet)
vec = tfidf.transform([clean])
pred = model.predict(vec)
label = le.inverse_transform(pred)[0]
print(f"Sentiment: {label.upper()}")
```

---

## Technologies Used

- **Python 3.13**
- **Data Manipulation**: Pandas, NumPy
- **NLP**: NLTK, spaCy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Machine Learning**: Scikit-learn (LinearSVC, MultinomialNB, RandomForest, TF-IDF)
- **Model Persistence**: Joblib
- **Development**: Jupyter Notebook

---

## Future Improvements

- [ ] Add word embeddings (Word2Vec, GloVe) for richer text representation
- [ ] Try deep learning models (LSTM, BERT) for better context understanding
- [ ] Address class imbalance with oversampling (SMOTE) or class weights
- [ ] Add real-time Twitter API integration for live sentiment monitoring
- [ ] Build Streamlit demo for interactive prediction
- [ ] Deploy model as REST API for production use

---

## Lessons Learned

- **Linear models often outperform complex models on text data** — LinearSVC beat RandomForest and Ensemble methods
- **Class imbalance matters** — negative tweets dominate (62.7%), so F1-score macro is more informative than accuracy
- **Neutral sentiment is hardest to classify** — it shares features with both positive and negative
- **TF-IDF with bigrams captures enough context** — no need for more complex embeddings for this dataset
- **Preprocessing helps but doesn't solve everything** — text length reduced 21.2%, but semantic understanding still limited

---

## Acknowledgments

- **Dataset**: [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) by Crowdflower
- **Inspiration**: The need for automated sentiment analysis in customer service
- **Tools**: Built with scikit-learn, NLTK, and the Python data science ecosystem

---

## Author

**Bagus Rahmadani**
- GitHub: [@Bagus2510](https://github.com/Bagus2510)
- LinkedIn: [Bagus Rahmadani](https://www.linkedin.com/in/bagusrahmadani/)
- Email: bagusrajin465@gmail.com

---

<div align="center">

**Made with ❤️ for Better Customer Service**

</div>
