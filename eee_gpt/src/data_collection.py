import pdfplumber

with pdfplumber.open('data/raw/textbook.pdf') as pdf:
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''

with open('data/raw/textbook.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("---------------------------------")
import requests
from bs4 import BeautifulSoup

url = 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text(separator=' ', strip=True)

with open('data/raw/notes.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("---------------------------------")


import os

raw_dir = 'data/raw'
all_text = ''
for filename in os.listdir(raw_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
            all_text += f.read() + '\n'

# Basic cleaning (e.g., remove extra whitespace)
all_text = ' '.join(all_text.split())

with open('data/processed/full_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(all_text)
with open('data/processed/full_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

split_idx = int(0.9 * len(text))
train_text = text[:split_idx]
val_text = text[split_idx:]

with open('data/processed/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_text)
with open('data/processed/val.txt', 'w', encoding='utf-8') as f:
    f.write(val_text)