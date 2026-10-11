# Downloads the public BBC News dataset into data/bbc_news.csv
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/suraj-deshmukh/"
       "BBC-Dataset-News-Classification/master/dataset/dataset.csv")

os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve(URL, "data/bbc_news.csv")
print("Saved data/bbc_news.csv")