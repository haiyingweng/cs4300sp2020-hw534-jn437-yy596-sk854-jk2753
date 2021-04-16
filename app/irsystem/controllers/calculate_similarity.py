import collections
import json
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import re

with open("../../../reviews.json", "r") as f:
    all_reviews = json.load(f)

with open("../../../tcin_cereal.json", "r") as t:
    tcin_to_cereal = json.load(t)

tcins = all_reviews.keys()


def tokenize_reviews(tokenizer, cereal_reviews):
    # Returns dictionary {tcin1: [token1, token2], tcin2: [token1, token3]}
    res = collections.defaultdict(list)
    for tcin, reviews in cereal_reviews.items():
        reviews = reviews["reviews"]
        for review in reviews:
            review_text = review["ReviewText"]
            tokens = tokenizer.tokenize(review_text.lower())
            res[tcin] += tokens
    return res


def get_reviews_vocab(tokenized_reviews):
    # Returns list containing vocabulary of [tokenized_reviews]
    tok_dict = tokenized_reviews
    vocab = set()
    for _, toks in tok_dict.items():
        vocab |= set(toks)
    return vocab


tokenizer = TreebankWordTokenizer()
tokenized_reviews = tokenize_reviews(tokenizer, all_reviews)
reviews_vocab = list(get_reviews_vocab(tokenized_reviews))
word_to_index = {word: i for i, word in enumerate(reviews_vocab)}
tcin_to_index = {tcin: i for i, tcin in enumerate(tcins)}


def get_tf(tokenized_reviews, reviews_vocab):
    # (i,j): number of times j appears in reviews for cereal i
    tf_matrix = np.zeros((len(tokenized_reviews), len(reviews_vocab)))
    for i, tcin in enumerate(tcins):
        words = tokenized_reviews[tcin]
        for word in words:
            word_index = word_to_index[word]
            tf_matrix[i][word_index] += 1
    return tf_matrix


def rank_by_similarity(query, tf_matrix, reviews_vocab):
    query_tokens = re.findall("[a-zA-Z]+", query.lower())
    cols = []
    for tok in query_tokens:
        if tok in reviews_vocab:
            cols.append(word_to_index[tok])
    matrix_cols = tf_matrix[:, cols]
    sum_cols = np.sum(matrix_cols, axis=1)
    score_lst = [(tcin_to_cereal[tcin], sum_cols[i]) for i, tcin in enumerate(tcins)]
    score_lst.sort(key=lambda x: x[1], reverse=True)
    print(score_lst)
    return score_lst


tf_matrix = get_tf(tokenized_reviews, reviews_vocab)
rank_by_similarity("almond", tf_matrix, reviews_vocab)
# def get_inverted_index():
