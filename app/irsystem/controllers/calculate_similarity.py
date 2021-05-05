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
cereal_to_tcin = {cereal: tcin for tcin, cereal in tcin_to_cereal.items()}
tcin_to_index = {tcin: i for i, tcin in enumerate(tcins)}
index_to_tcin = {i: tcin for i, tcin in enumerate(tcins)}


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
        cereal_info[tcin]["top_reviews"] = cereal_descriptions[tcin]["product"][
            "top_reviews"
        ]
        cereal_info[tcin]["cal"] = info["cal"]
        cereal_info[tcin]["pro"] = info["pro"]
        cereal_info[tcin]["fat1"] = info["fat1"]
        cereal_info[tcin]["sod"] = info["sod"]
        cereal_info[tcin]["sug"] = info["sug"]
        cereal_info[tcin]["carb"] = info["carb"]
        cereal_info[tcin]["fib"] = info["fib"]
        cereal_info[tcin]["pot"] = info["pot"]
        cereal_info[tcin]["veg"] = info["veg"]
        cereal_info[tcin]["gf"] = info["gf"]
        cereal_info[tcin]["pf"] = info["pf"]
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

idf_word_to_index = {word: i for i, word in enumerate(idf.keys())}


def get_tf_idf_matrix(inverted_index, idf):
    matrix = np.zeros((len(tcin_to_index), len(idf)))
    for tok in idf.keys():
        for tcin, tf in inverted_index[tok]:
            cereal_index = tcin_to_index[tcin]
            keyword_index = idf_word_to_index[tok]
            matrix[cereal_index][keyword_index] = tf
    tok_idfs = np.zeros(len(idf))
    for word, i in idf_word_to_index.items():
        tok_idfs[i] = idf[word]
    matrix = matrix / tok_idfs
    return matrix


def filteritems(request):
    filter_keys = [
        "cal",
        "pro",
        "fat1",
        "sod",
        "fib",
        "carb",
        "sug",
        "pot",
        "veg",
        "pf",
        "gf",
    ]
    filters = {key: [] for key in filter_keys}
    # 1
    if request.args.get("calcheckbox1"):
        # check if low calories is checked
        filters["cal"].append("LOW")
    if request.args.get("calcheckbox2"):
        # check if medium calories is checked
        filters["cal"].append("MEDIUM")
    if request.args.get("calcheckbox3"):
        # check if high calories is checked
        filters["cal"].append("HIGH")
    # 2
    if request.args.get("procheckbox1"):
        # check if low protein is checked
        filters["pro"].append("LOW")
    if request.args.get("procheckbox2"):
        # check if medium protein is checked
        filters["pro"].append("MEDIUM")
    if request.args.get("procheckbox3"):
        # check if high protein is checked
        filters["pro"].append("HIGH")
    # 3
    if request.args.get("fatcheckbox1"):
        # check if low Fat is checked
        filters["fat1"].append("LOW")
    if request.args.get("fatcheckbox2"):
        # check if medium Fat is checked
        filters["fat1"].append("MEDIUM")
    if request.args.get("fatcheckbox3"):
        # check if high Fat is checked
        filters["fat1"].append("HIGH")
    # 4
    if request.args.get("sodcheckbox1"):
        # check if low Sodium is checked
        filters["sod"].append("LOW")
    if request.args.get("sodcheckbox2"):
        # check if medium Sodium is checked
        filters["sod"].append("MEDIUM")
    if request.args.get("sodcheckbox3"):
        # check if high Sodium is checked
        filters["sod"].append("HIGH")
    # 5
    if request.args.get("fibcheckbox1"):
        # check if low Fiber is checked
        filters["fib"].append("LOW")
    if request.args.get("fibcheckbox2"):
        # check if medium Fiber is checked
        filters["fib"].append("MEDIUM")
    if request.args.get("fibcheckbox3"):
        # check if high Fiber is checked
        filters["fib"].append("HIGH")
    # 6
    if request.args.get("carbcheckbox1"):
        # check if low Carbohydrate is checked
        filters["carb"].append("LOW")
    if request.args.get("carbcheckbox2"):
        # check if medium Carbohydrate is checked
        filters["carb"].append("MEDIUM")
    if request.args.get("carbcheckbox3"):
        # check if high Carbohydrate is checked
        filters["carb"].append("HIGH")
    # 7
    if request.args.get("sugcheckbox1"):
        # check if low Sugar is checked
        filters["sug"].append("LOW")
    if request.args.get("sugcheckbox2"):
        # check if medium Sugar is checked
        filters["sug"].append("MEDIUM")
    if request.args.get("sugcheckbox3"):
        # check if high Sugar is checked
        filters["sug"].append("HIGH")
    # 8
    if request.args.get("potcheckbox1"):
        # check if low Potassium is checked
        filters["pot"].append("LOW")
    if request.args.get("potcheckbox2"):
        # check if medium Potassium is checked
        filters["pot"].append("MEDIUM")
    if request.args.get("potcheckbox3"):
        # check if high Potassium is checked
        filters["pot"].append("HIGH")

    if request.args.get("vegcheckbox"):
        # check if vegan is checked
        filters["veg"].append("TRUE")
    else:
        filters["veg"].append("FALSE")
        filters["veg"].append("TRUE")
    if request.args.get("PFcheckbox"):
        # check if Peanut Free is checked
        filters["pf"].append("TRUE")
    else:
        filters["pf"].append("FALSE")
        filters["pf"].append("TRUE")
    if request.args.get("GFcheckbox"):
        # check if Gluten Free is checked
        filters["gf"].append("TRUE")
    else:
        filters["gf"].append("FALSE")
        filters["gf"].append("TRUE")
    return filters


def filter_tcin(filters, tcin):
    # print(filters)
    for k, v in filters.items():
        if not v:
            return False
        if cereal_details[tcin][k] not in v:
            return False
    return True


# def rank_by_similarity(query, inverted_index, idf, doc_norms, filters):
#     # Returns list of tuples (cereal name, score)
#     query_tokens = re.findall("[a-zA-Z]+", query.lower())
#     query_tokens = get_stems(query_tokens)
#     cereal_scores = {tcin: 0 for tcin in tcins if filter_tcin(filters, tcin)}
#     for tok in set(query_tokens):
#         if tok in idf.keys():
#             for tcin, tf in inverted_index[tok]:
#                 if tcin in cereal_scores.keys():
#                     cereal_scores[tcin] += tf * idf[tok]
#     # normalize
#     for tcin in cereal_scores.keys():
#         cereal_scores[tcin] = cereal_scores[tcin] / doc_norms[tcin_to_index[tcin]]
#     score_lst = [(tcin, score) for tcin, score in cereal_scores.items() if score > 0]
#     score_lst.sort(key=lambda tup: (-tup[1], tup[0]))
#     return score_lst


def get_cereal_details(ranked):
    dets = []
    for tcin, score in ranked:
        detail = cereal_details[tcin]
        detail["score"] = round(score, 3)
        detail["tcin"] = tcin
        dets.append(detail)
    return dets

def allrankings():
    data = [cereal for cereal in cereal_details]
    return sorted(data, key = lambda c: c["rating"])

def veganranking():
    data = [cereal for cereal in cereal_details if cereal['vegan'] == "TRUE"]
    return sorted(data, key = lambda c: c["rating"])

def pfranking():
    data = [cereal for cereal in cereal_details if cereal['pf'] == "TRUE"]
    return sorted(data, key = lambda c: c["rating"])

def gfranking():
    data = [cereal for cereal in cereal_details if cereal['gf'] == "TRUE"]
    return sorted(data, key = lambda c: c["rating"])


tf_idf_matrix = get_tf_idf_matrix(inverted_index, idf)


def rocchio_update(query_tokens, relev, tf_idf_matrix, a=10, b=0.01):
    query_toks_counter = collections.Counter(query_tokens)
    q0 = np.zeros(len(idf))

    if len(relev) != 0:
        relev_indeces = [tcin_to_index[tcin] for tcin in relev]
        rel_max = np.max(tf_idf_matrix[relev_indeces], axis=0)
        for tok, _ in query_toks_counter.items():
            idx = idf_word_to_index[tok]
            q0[idx] = rel_max[idx]
        sum_rel = np.sum(tf_idf_matrix[relev_indeces], axis=0)
        q1 = a * len(relev) * q0 + b / len(relev) * sum_rel
    else:
        max_tf_idf = np.max(tf_idf_matrix, axis=0)
        for tok, _ in query_toks_counter.items():
            idx = idf_word_to_index[tok]
            q0[idx] = max_tf_idf[idx]
        q1 = a * q0
    q1 = np.clip(q1, a_min=0, a_max=None)
    return q1


def ranking_rocchio(query, tf_idf_matrix, filters, input_rocchio=rocchio_update):
    # get tokens
    query_tokens = re.findall("[a-zA-Z]+", query.lower())
    query_tokens = get_stems(query_tokens)
    query_tokens = [tok for tok in query_tokens if tok in idf_word_to_index]
    if not query_tokens:
        return []
    # get relevant cereals
    relev_names = get_cereals_for_keywords(query_tokens)
    relev = [cereal_to_tcin[cereal_name] for cereal_name in relev_names]
    # get cos sim
    q1 = input_rocchio(query_tokens, relev, tf_idf_matrix)
    numerator = np.dot(tf_idf_matrix, q1)
    demonin = (np.linalg.norm(q1)) * (np.linalg.norm(tf_idf_matrix, axis=1))
    sim = numerator / demonin
    # rank
    cereal_score_list = [
        (index_to_tcin[i], score)
        for i, score in enumerate(sim)
        if filter_tcin(filters, index_to_tcin[i]) and round(score, 3) != 0
    ]
    cereal_score_list = list(cereal_score_list)
    # sort cereal by score
    cereal_score_list.sort(key=lambda x: -x[1])

    return cereal_score_list[:15]


# def ranking_similar_cereals(query_cereal, tf_idf_matrix, filters):
#     if query_cereal in cereal_to_tcin:
#         query_tcin = cereal_to_tcin[query_cereal]
#         query_idx = tcin_to_index[query_tcin]
#         query = tf_idf_matrix[query_idx]
#         numerator = np.dot(tf_idf_matrix, query)
#         demonin = (np.linalg.norm(query)) * (np.linalg.norm(tf_idf_matrix, axis=1))
#         cos_sim = numerator / demonin
#         rank_list = [
#             (index_to_tcin[i], score)
#             for i, score in enumerate(cos_sim)
#             if filter_tcin(filters, index_to_tcin[i])
#         ]
#         rank_list = list(rank_list)
#         rank_list.sort(key=lambda x: -x[1])
#         return rank_list
#     else:
#         return []


# similar = ranking_similar_cereals("Lucky Charms", tf_idf_matrix, [])
# dets = []
# for tcin, score in similar:
#     detail = cereal_details[tcin]
#     detail["score"] = score
#     dets.append(detail["name"])
# print(dets)


# rocchio = ranking_rocchio("happy kid", tf_idf_matrix)
# dets = []
# for tcin, score in rocchio:
#         detail = cereal_details[tcin]
#         detail["score"] = score
#         dets.append(detail['name'])
# print(dets)

# query = "marshmallows"
# ranked_cereals = rank_by_similarity(query, inverted_index, idf, norms)
# ranked_cereal_details = get_cereal_details(ranked_cereals)
# print(ranked_cereal_details)
