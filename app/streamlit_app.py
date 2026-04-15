import streamlit as st
import joblib
import sys
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Add project root to path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.preprocess import clean_text

# Model directory
model_dir = os.path.join(project_root, "models")

# Load models
vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
lr_model = joblib.load(os.path.join(model_dir, "lr_model.pkl"))
svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
lda_model = joblib.load(os.path.join(model_dir, "lda_model.pkl"))

sia = SentimentIntensityAnalyzer()

st.title("🎬 Movie Review NLP Analyzer")

st.write("Analyze movie reviews using Sentiment Analysis, Keyword Extraction, and Topic Modeling.")

review = st.text_area("Enter Movie Review")

if st.button("Analyze Review"):

    if review:

        clean_review = clean_text(review)

        # TFIDF
        X = vectorizer.transform([clean_review])

        # Predictions
        lr_pred = lr_model.predict(X)[0]
        svm_pred = svm_model.predict(X)[0]

        # VADER sentiment
        vader_score = sia.polarity_scores(review)
        vader_sentiment = "Positive" if vader_score["compound"] > 0 else "Negative"

        st.subheader("Sentiment Prediction")

        col1, col2, col3 = st.columns(3)

        col1.metric("Logistic Regression", lr_pred)
        col2.metric("SVM", svm_pred)
        col3.metric("VADER", vader_sentiment)

        # Keyword Extraction
        r = Rake()
        r.extract_keywords_from_text(review)
        keywords = r.get_ranked_phrases()[:10]

        st.subheader("Extracted Keywords")

        st.write(keywords)

        # Topic Modeling
        topic_probs = lda_model.transform(X)
        topic = topic_probs.argmax()

        topic_labels = {
            0: "Martial Arts / Action Movies",
            3: "Cartoon / Animation Movies",
            4: "Classic / Old Movies"
        }

        st.subheader("Detected Topic")

        if topic in topic_labels:
            st.success(topic_labels[topic])
        else:
            st.info("General Movie Review")

        # WordCloud
        st.subheader("Word Cloud")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(review)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

    else:
        st.warning("Please enter a review")