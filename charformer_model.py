import tensorflow as tf
import random
import numpy as np
class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
        self.word_embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=output_dim,
        )
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.output_dim = output_dim
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "output_dim": self.output_dim
        })
        return config

def make_charformer(seq_len,vocsize,emb_dim,layers,heads,hmul):
    inl = tf.keras.layers.Input((seq_len,))
    nb = PositionEmbeddingLayer(seq_len,vocsize+1,emb_dim)(inl)
    l = nb

    for layer in range(layers):

        s = l
        l = tf.keras.layers.LayerNormalization()(l)
        dr = 1
        # https://www.researchgate.net/publication/334242716/figure/fig1/AS:777158343528450@1562300396701/Causal-dilated-convolution-with-kernel-size-2-and-dilation-factors-1-2-4-and-8.png
        while dr <= seq_len:
            l = tf.keras.layers.Conv1D(emb_dim,2,1,"causal",dilation_rate=dr,activation="gelu")(l)
            l = tf.keras.layers.Dropout(0.2)(l)
            dr *= 2
        l = l+s
        s = l
        l = tf.keras.layers.LayerNormalization()(l)
        q,k,v = tf.keras.layers.Dense(emb_dim,"gelu")(l),tf.keras.layers.Dense(emb_dim,"gelu")(l),tf.keras.layers.Dense(emb_dim,"gelu")(l)

        l = tf.keras.layers.MultiHeadAttention(heads,(emb_dim//heads)*hmul)(q,k,v,use_causal_mask=True)
        l = l+s
        s = l

        l = tf.keras.layers.LayerNormalization()(l)
        l = tf.keras.layers.Dense(emb_dim,"gelu")(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.Dense(emb_dim,"gelu")(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.Dense(emb_dim,"gelu")(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = l+s
    l = tf.keras.layers.LayerNormalization()(l)
    out = tf.keras.layers.Dense(vocsize,activation="softmax")(l)
    model = tf.keras.Model(inl,out)
    return model
def sample(probs, temperature=0.3):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    
    # Adjust probabilities with temperature
    adjusted_probs = np.power(probs, 1.0 / temperature)
    
    # Renormalize the probabilities
    renormalized_probs = adjusted_probs / np.sum(adjusted_probs)
    
    # Sample from the renormalized probabilities
    choices = np.arange(len(probs))
    sampled_index = np.random.choice(choices, p=renormalized_probs)
    
    return sampled_index
