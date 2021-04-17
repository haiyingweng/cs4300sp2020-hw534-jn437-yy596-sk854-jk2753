import json
import requests
from os import environ
from time import sleep

DETAILS_URL = "https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/details"

tcins = {}
inserted_data = {}


def get_description(tcin):
    print("get description for " + tcin)
    query_string = {"store_id": "3991", "tcin": tcin}
    headers = {
        "x-rapidapi-key": environ["RAPIDAPI_KEY"],
        "x-rapidapi-host": environ["RAPIDAPI_HOST"],
    }
    response = requests.request(
        "GET", DETAILS_URL, headers=headers, params=query_string
    )
    response_json = response.json()
    if tcin not in descriptions and tcin not in inserted_data:
        inserted_data[tcin] = response_json
    sleep(3)


with open("descriptions.json") as f:
    descriptions = json.load(f)

for tcin in tcins.keys():
    get_description(tcin)

descriptions.update(inserted_data)

with open("descriptions.json", "w") as f:
    json.dump(descriptions, f)
