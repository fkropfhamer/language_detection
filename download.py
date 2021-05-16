import requests


url = "https://downloads.tatoeba.org/exports/sentences.csv"

with requests.get(url, allow_redirects=True) as response:
  with open('data/sentences.csv', 'wb') as file:
    file.write(response.content)
