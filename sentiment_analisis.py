import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import string
import re

def preprocessText(textArray):
    lowercase = tf.strings.lower(textArray)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(string.punctuation)}]", ""
    )

data = pd.read_csv('./politifact_clean.csv/politifact_clean.csv')
print(data.describe())

DATASET_SIZE = len(data['veracity'].to_list())

labeldata = data.pop('veracity')
text_data = data.pop('statement')

dictionary = { "Pants on Fire!": 0, "False" : 1, "Mostly False" : 2, "Half-True": 3, "Mostly True" : 4, "True" : 5}
labeldata.replace(dictionary, inplace=True)

text_ds = tf.data.Dataset.from_tensor_slices((preprocessText(text_data.values), tf.one_hot(labeldata.values, depth=5)))

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = text_ds.shuffle(buffer_size=10)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = full_dataset.skip(val_size)
test_dataset = full_dataset.take(test_size)

num_features = 20000
sequence_length = 500
embbeded_dimensionality = 128


vectorize_layer = layers.TextVectorization(
standardize=preprocessText,
max_tokens=num_features,
output_mode="int",
output_sequence_length=sequence_length,)

vectorize_layer.adapt(text_data)


def vectorize_text(text, label):
    print(label)
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = train_dataset.map(vectorize_text)
val_ds = val_dataset.map(vectorize_text)
test_ds = test_dataset.map(vectorize_text)

for i in train_ds:
    print(i)
    break
inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(num_features + 1, embbeded_dimensionality)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.LSTM(128,return_sequences=True, activation='sigmoid')(x)
x = layers.Dropout(0.5)(x)
x = layers.LSTM(128,return_sequences=True,  activation='sigmoid')(x)
x = layers.Dropout(0.5)(x)
x = layers.LSTM(128, activation='sigmoid')(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

model.summary()

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# for x, y in text_ds.take(1):
#     print("Input:", x)
#     print("Target:", y)
# print(cleanTrainx)


epochs = 10

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.evaluate(test_ds)