import numpy as np
import tensorflow as tf
import sys

from sklearn.neighbors import KNeighborsClassifier

# set up mnist dataset
mnist = tf.keras.datasets.mnist
if len(sys.argv) == 2:
    command = sys.argv[1]
    if command == "original":
        pass
    elif command == "fashion":
        mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the size of training and test data
print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

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

k_means = KNeighborsClassifier()
k_means.fit(x_train_vector, y_train)

y_pred = k_means.predict(x_test_vector)

class_acc(y_pred, y_test)