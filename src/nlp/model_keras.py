import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_text_model(vocab_size=20000, max_len=200):
    return keras.Sequential([
        layers.Input(shape=(max_len,), dtype=tf.int32),
        layers.Embedding(vocab_size, 128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
