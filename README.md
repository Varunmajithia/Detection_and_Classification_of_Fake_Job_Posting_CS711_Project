# Detection_and_Classification_of_Fake_Job_Posting_CS711_Project

#Detection and Classification of Fake Job Postings

This project aims to tackle the growing threat of fake job advertisements by building an end-to-end machine learning pipeline that can detect fraudulent job postings based on textual and categorical data. The solution integrates traditional NLP techniques, multiple machine learning models, and a *Streamlit web app* for real-time testing.

---

## 📂 Dataset

- *Source*: [Kaggle: Fake Job Postings Dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
- *Records*: 17,880 job listings
- *Objective*: Predict the binary target fraudulent  
  (1 = Fake, 0 = Legitimate)

---

## 🔄 Pipeline Overview

### 🧹 Preprocessing Steps

All major text fields (title, company_profile, description, requirements, benefits) undergo the following transformations:

- Text lowercasing  
- HTML tag removal (BeautifulSoup)  
- Contraction expansion (e.g., “I'm” → “I am”)  
- Gibberish removal  
- Lemmatization  
- Stopword removal using NLTK

These steps help in standardizing and cleaning the data for downstream processing.

---

### 🛠 Feature Engineering

#### 📘 Textual Features

- *TF-IDF Vectors*:  
  Applied for *Logistic Regression, **Random Forest, and **Naive Bayes*

- *GloVe Word Embeddings*:  
  Applied for *XGBoost* to capture deeper word semantics

#### 🧮 Categorical Features

- *Target Encoding*:  
  Used with *Logistic Regression, **Naive Bayes, **XGBoost*

- *One-Hot Encoding*:  
  Applied specifically for *Random Forest*

#### 🔢 Numerical Features

- *MinMax Scaling*:  
  Ensures numerical features are scaled uniformly across all models

---

### 🧪 Feature Selection Techniques

- *Mutual Information*:  
  Selected informative features for Logistic Regression and Naive Bayes

- *Chi-Square Test*:  
  Used for categorical variables in Naive Bayes

- *SelectKBest*:  
  Applied to Random Forest for dimensionality reduction

---

## 🤖 Models Trained

| Model                | Text Representation | Categorical Encoding | Feature Selection     |
|---------------------|---------------------|----------------------|-----------------------|
| Logistic Regression | TF-IDF              | Target Encoding      | Mutual Information    |
| Random Forest       | TF-IDF              | One-Hot Encoding     | SelectKBest           |
| Naive Bayes         | TF-IDF              | Target Encoding      | MI + Chi-Square       |
| XGBoost             | GloVe Embeddings    | Target Encoding      | None                  |

Each model was tuned and tested independently to ensure robustness and generalizability.

---

## 📊 Evaluation Metrics

Models were evaluated using the following performance metrics:

- ✅ *Accuracy*
- 🟢 *Precision*
- 🔄 *Recall*
- ⭐ *F1-Score*
- 🧮 *Confusion Matrix*
- 🔁 *Cross-Validation* (GridSearchCV applied to Random Forest)

---

## 🌐 Interactive Streamlit App

A *Streamlit-based web application* was developed for easy, real-time interaction. Users can input job descriptions and receive immediate predictions on whether the posting is likely *real* or *fake*.

### ▶ To Launch the App Locally:

```bash
streamlit run app.py
