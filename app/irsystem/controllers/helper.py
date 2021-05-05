from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()


def get_stems(input):
    # Returns list of stems, [input] is of type list
    return [stemmer.stem(word.lower()) for word in input]


def stem_sentence(sentence):
    token_words = re.findall("[a-zA-Z]+", sentence.lower())
    stemmed = get_stems(token_words)
    return " ".join(stemmed)
