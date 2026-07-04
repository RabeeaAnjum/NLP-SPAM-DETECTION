# SMS Spam Detection Using Natural Language Processing

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Rabeea Anjum |
| **Course** | Natural Language Processing |
| **Department** | Computer Systems Engineering |
| **University** | The Islamia University of Bahawalpur |
| **Submitted To** | Dr. Nadia Rasheed |

---

## Project Description

This project implements a complete **Natural Language Processing (NLP) pipeline** for binary SMS Spam Detection using Python and Machine Learning. The system classifies SMS messages into two categories:

- **Spam** — Unsolicited, fraudulent, or promotional messages
- **Ham** — Legitimate, genuine personal messages

The project demonstrates real-world application of text preprocessing, TF-IDF feature extraction, and Multinomial Naive Bayes classification on the SMS Spam Collection Dataset.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | SMS Spam Collection Dataset |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) |
| **Total Messages** | 5,572 |
| **Spam Messages** | 747 (13.4%) |
| **Ham Messages** | 4,825 (86.6%) |
| **Classification** | Binary (Spam / Ham) |

---

## NLP Pipeline

```
Raw SMS Text
     │
     ▼
┌─────────────────────────┐
│   1. Data Loading       │  ← Pandas DataFrame
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   2. Text Preprocessing │  ← Lowercase → Remove Specials
│                         │     → Tokenize → Stopwords
│                         │     → Porter Stemming
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   3. Feature Extraction │  ← TF-IDF Vectorizer
│                         │     max_features = 5,000
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   4. Label Encoding     │  ← ham=0, spam=1
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   5. Train-Test Split   │  ← 80% Train / 20% Test
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   6. Model Training     │  ← Multinomial Naive Bayes
└───────────┬─────────────┘
            │
     ▼
┌─────────────────────────┐
│   7. Evaluation         │  ← Accuracy, Precision,
│                         │     Recall, F1, Confusion Matrix
└─────────────────────────┘
```

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| **Language** | Python 3 |
| **Environment** | Google Colab |
| **Data Handling** | Pandas, NumPy |
| **NLP** | NLTK (Stopwords, Porter Stemmer) |
| **ML** | Scikit-learn (TF-IDF, Naive Bayes) |
| **Visualization** | Matplotlib, Seaborn |

---

## Project Structure

```
SMS_Spam_Detection/
│
├── NLP_Spam_Detection.ipynb      ← Main Jupyter Notebook (Google Colab)
├── spam.csv                      ← Dataset (download separately)
├── README.md                     ← This file
├── Report.docx                   ← Project Report (Word Document)
│
└── outputs/                      ← Generated output files
    ├── confusion_matrix.png      ← Confusion matrix heatmap
    └── class_distribution.png   ← Class distribution bar chart
```

---

## Installation

Install all required dependencies using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

Or in Google Colab (first notebook cell):

```python
!pip install nltk scikit-learn pandas numpy matplotlib seaborn
```

---

## How to Run

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `NLP_Spam_Detection.ipynb`
3. Run **Cell 1** to install libraries
4. Run **Cell 3** to download NLTK stopwords
5. Run **Cell 4** to upload `spam.csv` when prompted
6. Run all remaining cells sequentially (**Runtime → Run All**)

### Option B — Local Jupyter

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook NLP_Spam_Detection.ipynb
```

> **Note:** If running locally, comment out the Google Colab file upload cell (Cell 4) and ensure `spam.csv` is in the same directory as the notebook.

---

## Dataset Download

The `spam.csv` dataset is not included in this repository due to size constraints.

**Download from:**
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

After downloading, place `spam.csv` in the same directory as the notebook.

---

## Expected Results

| Metric | Expected Value |
|---|---|
| **Overall Accuracy** | 97% – 99% |
| **Ham Precision** | ~0.98 |
| **Ham Recall** | ~0.99 |
| **Spam Precision** | ~0.96 |
| **Spam Recall** | ~0.92 |
| **Weighted F1-Score** | ~0.98 |

---

## Sample Predictions

| # | Message | Prediction | Confidence |
|---|---|---|---|
| 1 | Congratulations! You won a free iPhone. Click now. | **Spam** | ~99% |
| 2 | Can we meet tomorrow for the project discussion? | **Ham** | ~98% |
| 3 | Claim your cash prize immediately. | **Spam** | ~97% |
| 4 | Please submit the assignment before Friday. | **Ham** | ~96% |
| 5 | Win free rewards by clicking this link. | **Spam** | ~98% |

---

## Key Concepts

### TF-IDF (Term Frequency – Inverse Document Frequency)

Assigns numerical weights to words based on their importance:

```
TF-IDF(t, d) = TF(t, d) × log [ N / (1 + DF(t)) ]
```

- **TF** — How often a term appears in a document
- **IDF** — How rare the term is across all documents
- High TF-IDF → word is important to that specific document

### Multinomial Naive Bayes

A probabilistic classifier based on Bayes' theorem:

```
c* = argmax P(c) × ∏ P(xᵢ | c)
```

- Fast training, low computational cost
- Well-suited for high-dimensional sparse text data
- Strong baseline for text classification tasks

---

## Preprocessing Example

```python
Input  : "FREE OFFER!! Call NOW to claim your prize worth $1000."
Output : "free offer call claim prize worth"
```

Steps applied:
1. Lowercase: `"free offer!! call now to claim your prize worth $1000."`
2. Remove specials: `"free offer  call now to claim your prize worth  "`
3. Tokenize: `["free", "offer", "call", "now", "to", "claim", "your", "prize", "worth"]`
4. Remove stopwords: `["free", "offer", "call", "claim", "prize", "worth"]`
5. Stem: `["free", "offer", "call", "claim", "prize", "worth"]`

---

## Future Improvements

- [ ] Deep Learning with LSTM / BiLSTM
- [ ] Transformer-based models (BERT, DistilBERT)
- [ ] Multilingual SMS support (Urdu, Arabic)
- [ ] Real-time Flask / FastAPI deployment
- [ ] Ensemble methods (Naive Bayes + SVM + Logistic Regression)

---

## References

1. Jurafsky, D., & Martin, J. H. — *Speech and Language Processing* (3rd ed.)
2. Bird, S., Klein, E., & Loper, E. — *Natural Language Processing with Python*, O'Reilly
3. Pedregosa, F., et al. — *Scikit-learn: Machine Learning in Python*, JMLR 2011
4. Almeida, T. A., & Gómez Hidalgo, J. M. — *SMS Spam Collection Dataset*, UCI ML Repository
5. NLTK Documentation — https://www.nltk.org/
6. Scikit-learn Documentation — https://scikit-learn.org/

---

## Author

**Rabeea Anjum**  
Department of Computer Systems Engineering  
The Islamia University of Bahawalpur  
Course: Natural Language Processing | Instructor: Dr. Nadia Rasheed
