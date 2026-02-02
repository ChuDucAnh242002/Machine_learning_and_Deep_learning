import numpy as np
import tensorflow as tf
import sys
from scipy.stats import multivariate_normal
from numpy.linalg import matrix_rank


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

classes = np.unique(y_train)
stats = {}

# Cal the mean and variance
for c in classes:
    x_c = x_train_vector[y_train == c]
    mu = np.mean(x_c, axis=0)
    sigma2 = np.cov(x_c, rowvar=False)
    # print(f"Matrix rank of covariance in class {c}: {matrix_rank(sigma2)}")
    stats[c] = [mu, sigma2]

def cal_probability(x, stats):
    bayes_ps = np.empty([x.shape[0], len(stats.keys())])  # (10000, 10)
    for c in stats.keys():
        mu = stats[c][0]
        sigma2 = stats[c][1]
        bayes_ps[:,c] = multivariate_normal.logpdf(x, mu, sigma2)
    return np.argmax(bayes_ps, axis=1)

y_pred = cal_probability(x_test_vector, stats)
class_acc(y_pred, y_test)

