# Downloads the public BBC News dataset into data/bbc_news.csv
from pipeline import DATA_FILE, download_data

download_data()
print("Saved", DATA_FILE)