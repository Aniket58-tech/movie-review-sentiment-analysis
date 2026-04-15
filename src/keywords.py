from rake_nltk import Rake


def extract_keywords(text):

    r = Rake()

    r.extract_keywords_from_text(text)

    keywords = r.get_ranked_phrases()

    return keywords


def extract_keywords_corpus(text_list):

    r = Rake()

    keywords = []

    for text in text_list:

        r.extract_keywords_from_text(text)

        keywords.extend(r.get_ranked_phrases())

    return keywords