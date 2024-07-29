import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from charformer_model import make_charformer
import json
unk_char = "🫃"# unk token. pregnant man emoji :3
eof_char = "🫄"# eof token. pregnant person emoji :3
text = """"""
tparq = pd.read_parquet('2013.parquet', engine='pyarrow') # location of data parquet. example: https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/data/CC-MAIN-2013-20/000_00000.parquet
print("loading text")
# replace this with your own data code
txts = []
for row in tqdm(tparq["text"]):
    txts.append(row.replace(eof_char,unk_char))
    #text += row+eof_char
text = eof_char.join(txts)


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
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_freq=save_every)


spb = len(list(range(0,len(text)-seq_len,step)))//bsize

def tgen(): # text data generator
    r = list(range(0,len(text)-seq_len,step))
    random.shuffle(r)
    for _ in r:
        x_tex = text[_:_+seq_len]
        y_tex = text[_+1:_+seq_len+1]
        x_tex = "".join([_ if _ in vocab else unk_char for _ in x_tex])
        y_tex = "".join([_ if _ in vocab else unk_char for _ in y_tex])
        yield (tf.cast(tf.convert_to_tensor([vocab.index(c) for c in x_tex]),tf.int32),tf.cast(tf.keras.utils.to_categorical(tf.convert_to_tensor([vocab.index(c) for c in y_tex]),vocsize),tf.int8))




model = make_charformer(seq_len,vocsize,emb_dim,layers,heads,hmul)
model.summary()
# 0.00005
model.compile(tf.keras.optimizers.Adam(0.00005),tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
try:
    model.load_weights(checkpoint_filepath)
except:
    pass # if it doesn't exist, which it won't first time
for epoch in range(6):
    dataset = tf.data.Dataset.from_generator(
        lambda: tgen(),
        output_signature=(tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),tf.TensorSpec(shape=(seq_len,vocsize), dtype=tf.int8))
    ).batch(bsize)
    model.fit(dataset,steps_per_epoch=spb, callbacks=[model_checkpoint_callback])