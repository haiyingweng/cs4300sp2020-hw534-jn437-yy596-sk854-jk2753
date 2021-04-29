from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def get_stems(input):
    # Returns list of stems, [input] is of type list
    return [stemmer.stem(word.lower()) for word in input]
