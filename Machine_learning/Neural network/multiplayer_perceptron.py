import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
#import matplotlib
#matplotlib.use('TkAgg')  # Use TkAgg backend instead of macOS-specific backend

# Generate a sine wave
t = np.arange(0, 10, 0.1)
y = np.sin(t)
plt.plot(t, y)
plt.title('Training data for regression y=f(t)')
plt.xlabel('Time')
plt.ylabel('y = sin(t)')
plt.grid(True, which='both')
plt.show()

# Construct a MPL

# Model sequential
model = Sequential()
# 1st hidden layer (we also need to tell the input dimension)
#   10 neurons, but you can change to play a bit
model.add(Dense(50, input_dim=1, activation='sigmoid'))
## 2nd hidden layer - YOU MAY TEST THIS
#model.add(Dense(10, activation='sigmoid'))
# Output layer
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='tanh'))

# Learning rate has huge effect 
keras.optimizers.SGD(lr=0.2)
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

tr_hist = model.fit(t, y, epochs=10, verbose=0)
plt.plot(tr_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['opetus'], loc='upper right')
plt.show()

from sklearn.metrics import mean_squared_error 
y_pred = model.predict(t)
print(y[1])
print(y_pred[1])
print(np.sum(np.absolute(np.subtract(y,y_pred)))/len(t))
print(np.square(np.subtract(y,y_pred)).mean())
print(len(y))
print(np.divide(np.sum(np.square(y-y_pred)),len(y)))
print('MSE=',mean_squared_error(y,y_pred))
plt.plot(t, y, label='y')
plt.plot(t, y_pred, label='y_pred')
plt.title('Training data (sine wave)')
plt.xlabel('Time')
plt.ylabel('y = sin(t)')
plt.grid(True, which='both')
plt.legend()
plt.show()