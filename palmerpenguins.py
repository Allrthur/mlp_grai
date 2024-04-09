import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Model

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load Dataset

train = pd.read_csv("dataset/phishing/train.csv")
test = pd.read_csv("dataset/phishing/test.csv")

y_train = train["Result"]
X_train = train.drop(columns=["Result"])

y_test = test["Result"]
X_test = test.drop(columns=["Result"])

# Create Model

model = tf.keras.Sequential([
    Flatten(input_shape=(len(X_train.columns),)),
    Dense(15, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

sgd = SGD(learning_rate=0.1)
adam = Adam(learning_rate=0.1)

model.compile(
  loss='binary_crossentropy', 
  optimizer=adam, 
  metrics=['accuracy','precision','recall','f1_score']
)

model.summary()
model.fit(X_train, y_train, batch_size = 4, epochs=500, verbose=1)
model.evaluate(X_train,  y_train, verbose=2)