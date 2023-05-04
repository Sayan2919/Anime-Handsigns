import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def getModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(63,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


filenames = ['stop.csv', 'ok.csv', 'one.csv', 'two.csv', 'room.csv', 'shamble.csv']
dfs = []
dataset = pd.DataFrame()
for filename in filenames:
    df = pd.read_csv(filename, header=None, index_col=None)
    print(df.shape)
    dataset = pd.concat([dataset, df], axis=0, ignore_index=True)

dataset=dataset.sample(frac=1)
X,Y = dataset.loc[:,0:62], dataset.loc[:,63:]
X,Y = X.values, Y.values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)

model = getModel()
y_train = y_train.ravel()
print(y_train.shape)
y_train = tf.keras.utils.to_categorical(y_train)
print(y_train.shape)
model.fit(X_train, y_train, epochs=500, use_multiprocessing=True)


y_pred = model.predict(X_test)
print(y_pred[0],y_test[0])
