import requests
from bs4 import BeautifulSoup
import time

MOVIE_URLS = [
    'https://subslikescript.com/movie/Harry_Potter_and_the_Sorcerers_Stone-241527',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Chamber_of_Secrets-295297',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Prisoner_of_Azkaban-304141',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Goblet_of_Fire-330373',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Order_of_the_Phoenix-373889',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Half-Blood_Prince-417741',
    'https://subslikescript.com/movie/Harry_Potter_and_the_Deathly_Hallows_Part_1-926084',
]

OUTPUT_FILE = 'hp_full_script.txt'

all_scripts = []

for url in MOVIE_URLS:
    print(f"Scraping: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        script_div = soup.find('div', class_='full-script')
        if not script_div:
            print(f"Could not find script on {url}")
            continue
        script_text = script_div.get_text(separator='\n').strip()
        all_scripts.append(script_text)
        print(f"Added script from {url} (length: {len(script_text)} chars)")
        time.sleep(1)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for script in all_scripts:
        f.write(script + '\n\n')

print(f"Saved all scripts to {OUTPUT_FILE} (total length: {sum(len(s) for s in all_scripts)} chars)") 