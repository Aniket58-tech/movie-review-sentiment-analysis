# 🎬 Movie Review Sentiment Analysis & Insights System

An end-to-end NLP-based system that analyzes movie reviews to extract sentiment, key insights, and underlying topics. Built to demonstrate real-world applications of Machine Learning and Natural Language Processing.

---

## 🚀 Problem Statement

Understanding customer sentiment from large volumes of text data is critical for improving recommendations and business decisions. This project analyzes movie reviews to:

- Classify sentiment (Positive / Negative)
- Extract important keywords
- Identify hidden topics/themes
- Provide real-time insights via an interactive dashboard

---

## 💡 Solution

Developed a machine learning pipeline combined with NLP techniques to process and analyze 50K+ movie reviews. The system provides:

- Multi-model sentiment prediction
- Topic modeling for theme discovery
- Keyword extraction for insights
- Interactive visualization using Streamlit

---

## 🛠️ Tech Stack

- **Language:** Python  
- **ML/NLP:** Scikit-learn, NLTK, RAKE  
- **Algorithms:** Logistic Regression, SVM, VADER  
- **Feature Engineering:** TF-IDF  
- **Topic Modeling:** LDA  
- **Visualization:** Matplotlib, WordCloud  
- **Deployment/UI:** Streamlit  

---

## ⚙️ Key Features

- 🔍 Sentiment classification with **85%+ accuracy**
- 🧠 Topic modeling using **LDA for theme detection**
- 🏷️ Keyword extraction using **RAKE algorithm**
- 📊 Real-time insights via **Streamlit dashboard**
- 🧩 Modular and scalable project structure

---

## 📊 Results & Impact

- Analyzed **50,000+ reviews**
- Achieved **85%+ accuracy** using ML models
- Identified common user trends:
  - Positive → visuals, acting
  - Negative → slow storyline, weak script
- Enables data-driven decision-making for recommendations

---

## 🖥️ Demo

👉 Live App: *(Add your Streamlit link here)*  
👉 GitHub Repo: *(This repo)*  

---

## 📂 Project Structure
Movie_Review_Insights/
│── app/ # Streamlit app
│── src/ # Core logic (preprocessing, models)
│── models/ # Saved models (.pkl)
│── data/ # Dataset (not uploaded)
│── requirements.txt
│── README.md

---

## ▶️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/Aniket58-tech/movie-review-sentiment-analysis.git

# Navigate
cd movie-review-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/app.py
