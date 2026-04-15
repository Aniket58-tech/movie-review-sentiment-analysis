from sklearn.decomposition import LatentDirichletAllocation


def train_lda_model(tfidf_matrix, n_topics=5):

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(tfidf_matrix)

    return lda


def display_topics(model, feature_names, num_words=10):

    topics = []

    for topic_idx, topic in enumerate(model.components_):

        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-num_words - 1:-1]
        ]

        topics.append(top_words)

    return topics