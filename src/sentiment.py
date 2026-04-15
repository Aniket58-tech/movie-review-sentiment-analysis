from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_vectorizer():
    """
    Create TF-IDF vectorizer
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=5
    )

    return vectorizer


def train_sentiment_models(X, y):
    """
    Train Logistic Regression and SVM models
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    lr_model = LogisticRegression(max_iter=200)
    svm_model = LinearSVC()

    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_pred)
    svm_acc = accuracy_score(y_test, svm_pred)

    print(f"Logistic Regression Accuracy: {lr_acc}")
    print(f"SVM Accuracy: {svm_acc}")

    return lr_model, svm_model