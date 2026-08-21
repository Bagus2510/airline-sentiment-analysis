# Panduan Menjalankan Project

## Prasyarat

- Python 3.10 atau lebih tinggi
- pip (package manager)
- Git
- [Kaggle account](https://www.kaggle.com/) untuk download dataset

---

## 1. Clone Repository

```bash
git clone https://github.com/username/twitter-airline-sentiment.git
cd twitter-airline-sentiment
```

---

## 2. Setup Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download Dataset

1. Buka [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
2. Download file `Tweets.csv`
3. Pindahkan ke folder `data/raw/`

Struktur folder setelah download:
```
data/
├── raw/
│   └── Tweets.csv
├── processed/
│   └── (akan diisi otomatis setelah preprocessing)
```

---

## 5. Jalankan Notebook

### Urutan notebook (wajib berurutan):

| No | Notebook | Fungsi | Estimasi Waktu |
|----|----------|--------|----------------|
| 1 | `01_eda.ipynb` | Exploratory Data Analysis | ~2 menit |
| 2 | `02_preprocessing.ipynb` | Text Preprocessing | ~5 menit |
| 3 | `03_feature_extraction.ipynb` | TF-IDF Feature Extraction | ~3 menit |
| 4 | `04_modeling.ipynb` | Model Training & Evaluation | ~10 menit |

### Cara menjalankan:

```bash
cd notebooks
jupyter notebook
```

Klik notebook sesuai urutan, lalu jalankan semua cell (`Kernel` → `Restart & Run All`).

---

## 6. Jalankan Streamlit Demo

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

Browser akan terbuka di `http://localhost:8501`

---

## Struktur Project

```
Twitter US Airline Sentiment/
├── data/
│   ├── raw/                    # Dataset asli
│   └── processed/              # Hasil preprocessing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_modeling.ipynb
├── models/                     # Model yang sudah di-save
│   ├── model_final.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── streamlit/
│   ├── app.py
│   └── requirements.txt
├── images/                     # Screenshot & visualisasi
├── docs/                       # Dokumentasi
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Troubleshooting

### Error: `ModuleNotFoundError`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: `FileNotFoundError: data/raw/Tweets.csv`
- Pastikan dataset sudah didownload dari Kaggle
- Pastikan file `Tweets.csv` ada di folder `data/raw/`

### Streamlit app tidak bisa load model
- Pastikan notebook `04_modeling.ipynb` sudah dijalankan
- File `model_final.pkl` harus ada di folder `models/`

### Error: `use_container_width` deprecated
- Warning ini dari library Plotly, bukan dari code
- Aman diabaikan (akan hilang setelah Plotly update)

---

## Hasil yang Diharapkan

Setelah menjalankan semua notebook:

1. **01_eda.ipynb** → Visualisasi distribusi sentimen, word frequency
2. **02_preprocessing.ipynb** → File `processed/clean_tweets.csv`
3. **03_feature_extraction.ipynb** → TF-IDF matrix (27,780 fitur)
4. **04_modeling.ipynb** → Model terbaik: LinearSVC (80.36% accuracy)

---

## Catatan

- Dataset adalah **public domain** dari Kaggle
- Random state = 43 digunakan di semua splitting & model
- Stop words **tidak dihapus** (penting untuk sentimen)
- SMOTE diterapkan untuk menangani class imbalance
