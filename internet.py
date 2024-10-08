#%%
import tensorflow as tf
import pandas as pd
from keras.layers import Flatten, Dense, Softmax
from keras.optimizers import SGD, Adam
from keras.models import Model

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# from imblearn.over_sampling import SMOTE, RandomOverSampler

import os

#%%
# Load Dataset
CLASS_COLUMN_NAME = "Attack_type"

dataset = pd.read_csv("dataset/internet/data.csv").dropna()

# Creating one hot encoding
dataset = pd.get_dummies(dataset, columns=["Attack_type", "proto", "service"], dtype=float)
CLASS_COLUMNS = [column for column in dataset.columns if column.startswith(f"{CLASS_COLUMN_NAME}_")]

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
    Dense(80, activation='sigmoid'),
    Dense(40, activation='sigmoid'),
    Dense(20, activation='sigmoid'),
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
model.fit(X_train, y_train, batch_size=4, epochs=100, verbose=1)
model.evaluate(X_test,  y_test)

#%%
preds = model.predict(X_test)

def proba_to_onehot(proba_list):
    return [1 if proba == max(proba_list) else 0 for proba in proba_list]

preds = [proba_to_onehot(item) for item in preds]

# print(preds)

acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")

print("acc:\t", acc)
print("prec:\t", prec)
print("rec:\t", rec)
print("f1:\t", f1)

# printing results into file
FILE_PATH = "results/internet.txt"

if not os.path.exists(FILE_PATH):
    print("creating results file")
    with open(FILE_PATH, mode="w") as file: file.write("experiment,acc,prec,rec,f1\n")

last_experiment_idx = 1
with open(FILE_PATH, mode='r') as file: 
    last_experiment_idx = len(file.readlines())

with open(FILE_PATH, mode='a') as file:
    file.write(f"{last_experiment_idx},{acc},{prec},{rec},{f1}\n")