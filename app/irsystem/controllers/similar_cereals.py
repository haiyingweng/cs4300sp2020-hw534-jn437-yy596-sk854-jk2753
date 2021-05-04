import collections
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from .calculate_similarity import *

all_cereals = tcin_to_cereal.values()


def build_vectorizer(max_n_terms=5000, max_prop_docs=0.8, min_n_docs=10):
    return TfidfVectorizer(
        stop_words="english",
        min_df=min_n_docs,
        max_df=max_prop_docs,
        max_features=max_n_terms,
    )


tfidf_vec = build_vectorizer()
tfidf_mat = tfidf_vec.fit_transform(
    [info["product"]["description"] for tcin, info in cereal_descriptions.items()]
).toarray()


def get_cosine_sim(cereal1, cereal2, tfidf_mat, tcin_to_index):

    indx1 = tcin_to_index[cereal1]
    indx2 = tcin_to_index[cereal2]
    top = np.dot(tfidf_mat[indx1], tfidf_mat[indx2])
    denominator = np.sqrt(np.sum(np.square(tfidf_mat[indx1]))) * np.sqrt(
        np.sum(np.square(tfidf_mat[indx2]))
    )
    return top / denominator


def build_cos_sim_matrix(num_cereals, tfidf_mat, input_get_sim_method=get_cosine_sim):
    res = np.ones((num_cereals, num_cereals))
    for i in range(len(res)):
        for j in range(i + 1, len(res[i])):
            sim = input_get_sim_method(
                index_to_tcin[i], index_to_tcin[j], tfidf_mat, tcin_to_index
            )
            res[i][j] = sim
            res[j][i] = sim
    return res


cereal_sims_cos = build_cos_sim_matrix(num_cereals, tfidf_mat)


def rank_by_similar_cereal(query_cereal, sim_matrix):
    tcin = cereal_to_tcin[query_cereal]
    idx = tcin_to_index[tcin]
    score_lst = sim_matrix[idx]
    score_lst = [(index_to_tcin[i], s) for i, s in enumerate(score_lst)]
    score_lst = score_lst[:idx] + score_lst[idx + 1 :]
    score_lst = sorted(score_lst, key=lambda x: -x[1])

    return score_lst[:15]


similar = rank_by_similar_cereal("Corn Pops", cereal_sims_cos)
dets = []
for tcin, score in similar:
    detail = cereal_details[tcin]
    detail["score"] = score
    dets.append(detail["name"])
print(dets)
