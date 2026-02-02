import matplotlib.pyplot as plt
import numpy as np
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras.optimizers
import keras


# set up mnist dataset
mnist = tf.keras.datasets.mnist
if len(sys.argv) == 2:
    command = sys.argv[1]
    if command == "original":
        pass
    elif command == "fashion":
        mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def class_acc(pred, gt):
    # pred: predicted
    # gt: groundtruth
    corr = np.count_nonzero(np.equal(pred, gt))
    success_rate = corr / y_test.shape[0]
    print(f"Classification accuracy is {success_rate:.2f}")

x_train_vector = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_vector = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Add noise to training vector
x_train_noise = np.random.normal(loc=0.0, scale=10, size=x_train_vector.shape)
x_train_vector = x_train_vector + x_train_noise

classes = np.unique(y_train)

y_train_hot = np.eye(len(classes))[y_train]

model = Sequential()
model.add(Dense(400, input_dim=784, activation='relu6'))
model.add(Dense(200, input_dim=400, activation='relu6'))
model.add(Dense(100, input_dim=200, activation='relu6'))
model.add(Dense(50, input_dim=100, activation='relu6'))
model.add(Dense(10, input_dim=50, activation='softmax'))

keras.optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

history = model.fit(x_train_vector, y_train_hot, epochs=10, verbose=1)
plt.plot(history.history['loss'], label="Training")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

y_test_pred_2 = model.predict(x_test_vector)
y_test_pred = np.argmax(y_test_pred_2, axis=1)

class_acc(y_test_pred, y_test)