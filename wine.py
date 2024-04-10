#%%
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Softmax
from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Model

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

#%%
# Load Dataset
CLASS_COLUMNS = ["origin_1", "origin_2", "origin_3"]
dataset = pd.read_csv("dataset/wine/wine.csv").dropna()
dataset["origin"] = dataset["origin"].astype(object)

# Creating one hot encoding
dataset = pd.get_dummies(dataset, columns=["origin"], dtype=float)

# Separating splits and X and y
train, test = train_test_split(dataset, train_size=0.7, shuffle=True)

y_train = train[CLASS_COLUMNS]
X_train = train.drop(columns=CLASS_COLUMNS)

y_test = test[CLASS_COLUMNS]
X_test = test.drop(columns=CLASS_COLUMNS)

#%%
# Create Model
model = tf.keras.Sequential([
    Flatten(input_shape=(len(X_train.columns),)),
    Dense(32, activation='sigmoid'),
    Dense(16, activation='sigmoid'),
    Dense(len(y_train.columns) if type(y_train)!=pd.Series else len(y_train), activation='softmax'),
])

sgd = SGD(learning_rate=0.1)
adam = Adam(learning_rate=1e-4)

model.compile(
  loss='crossentropy', 
  optimizer=adam, 
  metrics=['accuracy','precision','recall','f1_score']
)

model.summary()
model.fit(X_train, y_train, batch_size=4, epochs=500, verbose=1)
model.evaluate(X_test,  y_test)

#%%
preds = model.predict(X_test)

def proba_to_onehot(proba_list):
    return [1 if proba == max(proba_list) else 0 for proba in proba_list]

preds = [proba_to_onehot(item) for item in preds]

print(preds)

acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")

print("acc:\t", acc)
print("prec:\t", prec)
print("rec:\t", rec)
print("f1:\t", f1)