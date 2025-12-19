import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .preprocess_nltk import clean_text, tokenize_keep

def main():
    out_dir = "outputs/nlp"
    os.makedirs(out_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=20000)

    word_index = keras.datasets.imdb.get_word_index()
    rev_index = {v + 3: k for k, v in word_index.items()}
    rev_index[0], rev_index[1], rev_index[2] = "<PAD>", "<START>", "<UNK>"

    def decode(seq):
        return " ".join([rev_index.get(i, "<UNK>") for i in seq])

    raw_train = [tokenize_keep(clean_text(decode(s))) for s in x_train[:20000]]
    raw_test  = [tokenize_keep(clean_text(decode(s))) for s in x_test[:5000]]
    y_train_small = y_train[:20000]
    y_test_small  = y_test[:5000]

    vectorizer = layers.TextVectorization(
        max_tokens=20000,
        output_mode="int",
        output_sequence_length=200
    )
    vectorizer.adapt(raw_train)

    # Dataset now stays as STRINGS
    train_ds = tf.data.Dataset.from_tensor_slices((raw_train, y_train_small)).batch(64).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((raw_test,  y_test_small)).batch(128).prefetch(tf.data.AUTOTUNE)

    # Model takes string input and vectorizes internally
    text_in = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(text_in)
    x = layers.Embedding(20000, 128)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(text_in, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, validation_data=test_ds, epochs=3)

    model.save(f"{out_dir}/sentiment_with_vectorizer.keras")
    print("Saved:", f"{out_dir}/sentiment_with_vectorizer.keras")

if __name__ == "__main__":
    main()
 