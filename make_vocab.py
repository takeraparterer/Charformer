import pandas as pd
from tqdm import tqdm
from collections import Counter
import json
unk_char = "🫃"# unk token. pregnant man emoji :3
eof_char = "🫄"# eof token. pregnant person emoji :3
text = """"""
tparq = pd.read_parquet('2013.parquet', engine='pyarrow')
print("loading text")
txts = []
for row in tqdm(tparq["text"]):
    txts.append(row.replace(eof_char,unk_char))
    #text += row+eof_char
text = eof_char.join(txts)


min_occurences = 50# the minimum number of times a character must appear in text for it to be considered in vocab
print("getting full vocab")
vocab = list(set(text))
vocab.sort()
voc2 = [unk_char,eof_char]
print("reducing vocab")
chrcount = Counter(text)
for char in tqdm(vocab):
    if chrcount[char] > min_occurences:
        voc2.append(char)
vocab = voc2
print(vocab)
with open("vocab.json","w+") as f:
    json.dump(vocab,f)
    f.close()