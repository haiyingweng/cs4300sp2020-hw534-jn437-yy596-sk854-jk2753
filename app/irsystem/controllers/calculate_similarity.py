import collections
import csv
import json
import math
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import re
import os.path

from .db_related import *
from .helper import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

manufacturers = {
    "G": "General Mills",
    "K": "Kellogg's",
    "P": "Post",
    "Q": "Quaker",
    "N": "Nature's Path",
}

with open(os.path.join(BASE_DIR, "static/reviews.json"), "r") as f:
    all_reviews = json.load(f)

with open(os.path.join(BASE_DIR, "static/tcin_cereal.json"), "r") as t:
    tcin_to_cereal = json.load(t)

with open(os.path.join(BASE_DIR, "static/descriptions.json"), "r") as d:
    cereal_descriptions = json.load(d)

with open(os.path.join(BASE_DIR, "static/cereal.csv"), mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    cereal_nutritions = {}
    for row in csv_reader:
        cereal_nutritions[row["name"]] = row

tcins = all_reviews.keys()
num_cereals = len(tcins)


def process_cereal_details():
    cereal_info = collections.defaultdict(dict)
    for tcin, name in tcin_to_cereal.items():
        info = cereal_nutritions[name]
        cereal_info[tcin]["name"] = name
        cereal_info[tcin]["manufacturer"] = manufacturers[info["mfr"]]
        cereal_info[tcin]["calories"] = f"{info['calories']}cal"
        cereal_info[tcin]["protein"] = f"{info['protein']}g"
        cereal_info[tcin]["fat"] = f"{info['fat']}g"
        cereal_info[tcin]["sodium"] = f"{info['sodium']}mg"
        cereal_info[tcin]["sugars"] = f"{info['sugars']}g"
        cereal_info[tcin]["carbo"] = f"{info['carbo']}g"
        cereal_info[tcin]["fiber"] = f"{info['fiber']}g"
        cereal_info[tcin]["potass"] = f"{info['potass']}mg"
        cereal_info[tcin][
            "cups"
        ] = f"{info['cups']} cup{'s' if float(info['cups'])>1 else ''}"
        cereal_info[tcin]["rating"] = info["rating"]
        image = info["img_url"] = cereal_descriptions[tcin]["product"]["images"][0]
        cereal_info[tcin]["img_url"] = image["base_url"] + image["primary"]
        cereal_info[tcin]["description"] = cereal_descriptions[tcin]["product"][
            "description"
        ]
        cereal_info[tcin]["bullets"] = cereal_descriptions[tcin]["product"][
            "soft_bullets"
        ]["bullets"]
    return cereal_info


def tokenize_reviews(tokenizer, cereal_reviews):
    # Returns dictionary {tcin1: [token1, token2], tcin2: [token1, token3]}
    res = collections.defaultdict(list)
    for tcin, reviews in cereal_reviews.items():
        reviews = reviews["reviews"]
        for review in reviews:
            review_text = review["ReviewText"]
            tokens = tokenizer.tokenize(review_text.lower())
            res[tcin] += tokens
        res[tcin] = get_stems(res[tcin])
    return res


def get_reviews_vocab(tokenized_reviews):
    # Returns list containing vocabulary of [tokenized_reviews]
    tok_dict = tokenized_reviews
    vocab = set()
    for _, toks in tok_dict.items():
        vocab |= set(toks)
    return vocab


cereal_details = process_cereal_details()

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


def filteritems(request):
    filters = dict.fromkeys(["cal", "pro", "fat", "sod", "fib", "carb", "sug", "pot", "veg", "pf", "gf"], [])
    #1
    if request.form.get('calcheckbox1'):
    #check if low calories is checked
        filters["cal"] += "LOW"
    if request.form.get('calcheckbox2'):
    #check if medium calories is checked
        filters["cal"] += "MEDIUM"
    if request.form.get('calcheckbox3'):
    #check if high calories is checked
        filters["cal"] += "HIGH"
    #2
    if request.form.get('procheckbox1'):
    #check if low protein is checked
        filters["pro"] += "LOW"
    if request.form.get('procheckbox2'):
    #check if medium protein is checked
        filters["pro"] += "MEDIUM"
    if request.form.get('procheckbox3'):
    #check if high protein is checked
        filters["pro"] += "HIGH"
    #3
    if request.form.get('fatcheckbox1'):
    #check if low Fat is checked
        filters["fat1"] += "LOW"
    if request.form.get('fatcheckbox2'):
    #check if medium Fat is checked
        filters["fat1"] += "MEDIUM"
    if request.form.get('fatcheckbox3'):
    #check if high Fat is checked
        filters["fat1"] += "HIGH"
    #4
    if request.form.get('sodcheckbox1'):
    #check if low Sodium is checked
        filters["sod"] += "LOW"
    if request.form.get('sodcheckbox2'):
    #check if medium Sodium is checked
        filters["sod"] += "MEDIUM"
    if request.form.get('sodcheckbox3'):
    #check if high Sodium is checked
        filters["sod"] += "HIGH"
    #5
    if request.form.get('fibcheckbox1'):
    #check if low Fiber is checked
        filters["fib"] += "LOW"
    if request.form.get('fibcheckbox2'):
    #check if medium Fiber is checked
        filters["fib"] += "MEDIUM"
    if request.form.get('fibcheckbox3'):
    #check if high Fiber is checked
        filters["fib"] += "HIGH"
    #6
    if request.form.get('carbcheckbox1'):
    #check if low Carbohydrate is checked
        filters["carb"] += "LOW"
    if request.form.get('carbcheckbox2'):
    #check if medium Carbohydrate is checked
        filters["carb"] += "MEDIUM"
    if request.form.get('carbcheckbox3'):
    #check if high Carbohydrate is checked
        filters["carb"] += "HIGH"
    #7
    if request.form.get('sugcheckbox1'):
    #check if low Sugar is checked
        filters["sug"] += "LOW"
    if request.form.get('sugcheckbox2'):
    #check if medium Sugar is checked
        filters["sug"] += "MEDIUM"
    if request.form.get('sugcheckbox3'):
    #check if high Sugar is checked
        filters["sug"] += "HIGH"
    #8
    if request.form.get('potcheckbox1'):
    #check if low Potassium is checked
        filters["pot"] += "LOW"
    if request.form.get('potcheckbox2'):
    #check if medium Potassium is checked
        filters["pot"] += "MEDIUM"
    if request.form.get('potcheckbox3'):
    #check if high Potassium is checked
        filters["pot"] += "HIGH"

    if request.form.get('vegcheckbox'):
    #check if vegan is checked
        filters["veg"] += "TRUE"
    else: filters["veg"] += "FALSE"
    if request.form.get('PFcheckbox'):
    #check if Peanut Free is checked
        filters["pf"] += "TRUE"
    else: filters["pf"] += "FALSE"
    if request.form.get('GFcheckbox'):
    #check if Gluten Free is checked
        filters["gf"] += "TRUE"
    else: filters["gf"] += "FALSE"
    return filters

def filter(filters, tcin):
  for k,v in filters:
    if not v: return False
    if cereal_info[tcin][k] in v:
        pass
    else: return False
  return True

def rank_by_similarity(query, inverted_index, idf, doc_norms, filters):
    # Returns list of tuples (cereal name, score)
    query_tokens = re.findall("[a-zA-Z]+", query.lower())
    query_tokens = get_stems(query_tokens)
    cereal_scores = {tcin: 0 for tcin in tcins}
    for tok in set(query_tokens):
        if tok in idf.keys():
            for tcin, tf in inverted_index[tok]:
                cereal_scores[tcin] += tf * idf[tok]
    # normalize
    for tcin in cereal_scores.keys():
        if filter(filters, tcin):
            cereal_scores[tcin] = cereal_scores[tcin] / doc_norms[tcin_to_index[tcin]]
        else:
            pass
    score_lst = [
        (tcin_to_cereal[tcin], tcin, score)
        for tcin, score in cereal_scores.items()
        if score > 0
    ]
    score_lst.sort(key=lambda tup: (-tup[2], tup[0]))
    return score_lst


def get_cereal_details(ranked):
    dets = []
    for name, tcin, score in ranked:
        detail = cereal_details[tcin]
        detail["score"] = score
        dets.append(detail)
    return dets


# query = "marshmallows"
# ranked_cereals = rank_by_similarity(query, inverted_index, idf, norms)
# ranked_cereal_details = get_cereal_details(ranked_cereals)
# print(ranked_cereal_details)
