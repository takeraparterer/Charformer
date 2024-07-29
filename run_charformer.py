import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from charformer_model import make_charformer, sample
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# this file loads the model, displays embeddings using TSNE reduction, and generates some text

unk_char = "🫃"# unk token. pregnant man emoji :3
eof_char = "🫄"# eof token. pregnant person emoji :3
text = """"""
print("loading text")
# these must be the same as in main_charformer.py

emb_dim = 512 # embedding dimension
bsize = 8 # batch size
layers = 16 # transformer hybrid layers
heads = 16 # attention heads per layer
hmul = 2 # head dim multiplier. head dim = (emb_dim/heads) * hmul
save_every = 500 # how many steps to save the model
with open("vocab.json","r") as f: # you need to run make_vocab.py to generate this
    vocab = json.load(f)
    f.close()
print(vocab)

seq_len = 256 # model sequence length in characters to train
step = (seq_len//2)-1 # get a seq_len amount of data every this many characters
vocsize = len(vocab)
model_name = "charpred_fineweb_256.h5" # your model filename
checkpoint_filepath = f"models/{model_name}" # checkpoint folder


model = make_charformer(seq_len,vocsize,emb_dim,layers,heads,hmul)
model.summary()
# .635
# 0.00005
model.compile(tf.keras.optimizers.Adam(0.00005),tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.load_weights(checkpoint_filepath)

embedding_layer = model.layers[1]
embeddings = embedding_layer.get_weights()[0]
print(embeddings)

# Step 3: Use t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2,perplexity=50, random_state=42,n_iter=10000)
embeddings_2d = tsne.fit_transform(embeddings)



# Step 4: Plot the 2D embeddings
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
# Assuming you have a vocabulary list
vocabulary = vocab
only_to_show = list("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890~!@#$%^&*()-=_+[]{}\\|;':\",./<>?")
# Label each point with the corresponding word
v2e = {}
for i, word in enumerate(vocabulary):
    #if word in only_to_show:
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    v2e[word] = (float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1]))
with open("embeddings_2d_new.json","w+") as f:
    json.dump(v2e,f)
    f.close()
plt.title('2D Visualization of Embeddings Layer')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()





def generate_1_char_rpad(text):
    text = text[-seq_len:]
    lt = len(text)-1 # index of where generated character will be
    text += eof_char*(seq_len-len(text))
    mi = text
    mi = [vocab.index(_) for _ in mi]
    mi = tf.convert_to_tensor([mi])
    pred = model(mi)
    pred = pred.numpy()
    pred = pred[0][lt]
    pred = sample(pred)
    pred = vocab[pred]
    return pred
intex = "Unicorns are"
for i in range(512):
    mi = intex[-seq_len:]
    pred = generate_1_char_rpad(intex)
    print(pred,end="",flush=True)
    intex += pred