# !!!! READ THIS ABOUT RUNNING THIS SCRIPT:
# https://www.notion.so/Using-cereal_reviews-py-script-2aab2104306f467f9520b57e75ffbf69

import json
import requests
from os import environ
from time import sleep

REVIEWS_URL = "https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/reviews"

# Example: tcins = {"54446420": "Raisin Bran", "78364946": "Raisin Nut Bran"}
tcins = {}
inserted_data = {}


def get_reviews(tcin):
    print("get reviews for " + tcin)
    query_string = {"tcin": tcin, "limit": "100", "offset": "0"}
    headers = {
        "x-rapidapi-key": environ["RAPIDAPI_KEY"],
        "x-rapidapi-host": environ["RAPIDAPI_HOST"],
    }
    response = requests.request(
        "GET", REVIEWS_URL, headers=headers, params=query_string
    )
    response_json = response.json()
    if tcin not in reviews and tcin not in inserted_data:
        inserted_data[tcin] = response_json
    sleep(1)


with open("reviews.json") as f:
    reviews = json.load(f)

with open("tcin_cereal.json") as f:
    tcin_cereal = json.load(f)

# Remove duplicates from tcin_cereal that we already have reviews for:
new_dict = {}
for tcin, cereal in tcins.items():
    if tcin not in reviews:
        new_dict[tcin] = cereal
tcins = new_dict

for tcin in tcins.keys():
    get_reviews(tcin)

reviews.update(inserted_data)
tcin_cereal.update(tcins)

with open("reviews.json", "w") as f:
    json.dump(reviews, f)

with open("tcin_cereal.json", "w") as f:
    json.dump(tcin_cereal, f)
