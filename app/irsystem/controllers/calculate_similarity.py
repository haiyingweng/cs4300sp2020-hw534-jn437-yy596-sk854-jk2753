import collections
import json
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import re

with open("../../../reviews.json", "r") as f:
    all_reviews = json.load(f)

with open("../../../tcin_cereal.json", "r") as t:
    tcin_to_cereal = json.load(t)

tcins = all_reviews.keys()
num_cereals = len(tcins)


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


def get_inverted_index(tokenized_reviews):
    # Returns dict: keys = terms in reviews vocabulary. values = list of tuples (tcin, TF of term in that cereal's reviews)
    inv_idx = {}
    for tcin, toks in tokenized_reviews.items():
        toks_set = set(toks)
        for tok in toks_set:
            new_elt = [(tcin, toks.count(tok))]
            if tok in inv_idx:
                inv_idx[tok] += new_elt
            else:
                inv_idx[tok] = new_elt
    return inv_idx


def get_idf(inverted_index, n_docs, min_df=2, max_df_ratio=0.90):
    # Returns a dict for each term contains a value of idf value
    idf_dict = {}
    for key, value in inverted_index.items():
        doc_num = len(value)
        doc_ratio = doc_num / n_docs
        if doc_num >= min_df and doc_ratio <= max_df_ratio:
            idf_dict[key] = math.log(n_docs / (1 + doc_num), 2)
    return idf_dict


def get_doc_norms(inverted_index, idf, n_docs):
    # Returns np.array of size n_docs where norms[i]=norm of doc tcin_to_index[i]
    norms = np.zeros(n_docs)
    for word, postings in inverted_index.items():
        if word in idf:
            for doc, tf in postings:
                norms[tcin_to_index[doc]] += (tf * idf[word]) ** 2
    norms = np.sqrt(norms)
    return norms


inverted_index = get_inverted_index(tokenized_reviews)
idf = get_idf(inverted_index, num_cereals)
norms = get_doc_norms(inverted_index, idf, num_cereals)


def rank_by_similarity(query, inverted_index, idf, doc_norms):
    # Returns list of tuples (cereal name, score)
    query_tokens = re.findall("[a-zA-Z]+", query.lower())
    cereal_scores = {tcin: 0 for tcin in tcins}
    for tok in set(query_tokens):
        if tok in inverted_index:
            for tcin, tf in inverted_index[tok]:
                cereal_scores[tcin] += tf * idf[tok]
    # normalize
    for tcin in cereal_scores.keys():
        cereal_scores[tcin] = cereal_scores[tcin] / doc_norms[tcin_to_index[tcin]]
    score_lst = [
        (tcin_to_cereal[tcin], score)
        for tcin, score in cereal_scores.items()
        if score > 0
    ]
    score_lst.sort(key=lambda tup: (-tup[1], tup[0]))
    print(score_lst)
    return score_lst


stemmer = PorterStemmer()


def getstems(input):
    # Returns list of stems, [input] is of type list
    return [stemmer.stem(w.lower()) for word in input]


rank_by_similarity("fun", inverted_index, idf, norms)