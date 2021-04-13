import json
import requests
from os import environ

REVIEWS_URL = "https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/reviews"

product_tcins = []
inserted_data = {}


def get_reviews(tcin):
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


with open("reviews.json") as f:
    reviews = json.load(f)

for tcin in product_tcins:
    get_reviews(tcin)

reviews.update(inserted_data)

with open("reviews.json", "w") as f:
    json.dump(reviews, f)
