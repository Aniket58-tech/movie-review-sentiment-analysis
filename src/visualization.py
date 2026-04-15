import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(keyword_dict):

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(keyword_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Keyword WordCloud")

    plt.show()


def plot_topic_words(topic_words):

    plt.figure(figsize=(8, 5))

    plt.barh(topic_words, range(len(topic_words)))

    plt.title("Top Topic Words")

    plt.show()