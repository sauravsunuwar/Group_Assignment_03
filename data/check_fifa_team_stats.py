import requests

stats_id = 151600

url = f"https://fdh-api.fifa.com/v1/stats/match/{stats_id}/teams.json"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

data = response.json()

print("Status code:", response.status_code)
print("\n=== SHOT / ATTEMPT / XG RELATED FIELDS ===\n")


def search_stats(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            search_stats(value)

    elif isinstance(obj, list):

        # FIFA statistics appear as:
        # ["StatisticName", value, true]
        if (
            len(obj) >= 2
            and isinstance(obj[0], str)
            and any(
                word in obj[0].lower()
                for word in ["shot", "attempt", "xg"]
            )
        ):
            print(obj)

        for item in obj:
            search_stats(item)


search_stats(data)