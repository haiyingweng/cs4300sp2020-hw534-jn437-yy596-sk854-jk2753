from app.irsystem.models.keyword_cereal import *
import re

from .helper import *


def add_cereal_for_query(query, cereal):
    query_tokens = re.findall("[a-zA-Z]+", query.lower())
    query_tokens = get_stems(query_tokens)
    for tok in query_tokens:
        create_keyword_cereal(tok, cereal)


def get_cereals_for_keywords(keywords):
    all_cereals = []
    for keyword in keywords:
        cereals = KeywordCereal.query.filter_by(keyword=keyword)
        cereal_names = [cereal.cereal for cereal in cereals]
        all_cereals += cereal_names
    return set(all_cereals)


def create_keyword_cereal(keyword, cereal):
    keyword_cereal = KeywordCereal(keyword=keyword, cereal=cereal)
    db.session.add(keyword_cereal)
    db.session.commit()
    return keyword_cereal
