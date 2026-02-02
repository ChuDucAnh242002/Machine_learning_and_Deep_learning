import matplotlib.pyplot as plt
import numpy as np

import keras.optimizers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
# from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def class_acc(pred, gt):
    # pred: predicted
    # gt: groundtruth
    corr = np.count_nonzero(np.equal(pred, gt))
    success_rate = corr / gt.shape[0]
    print(f"Classification accuracy is {success_rate:.2f}")

np.random.seed(11) # to always get the same points
N = 200
x_h = np.random.normal(1.1,0.3,N)
x_e = np.random.normal(1.9,0.4,N)

N_t = 50
x_h_test = np.random.normal(1.1,0.3,N_t) # h as hobit
x_e_test = np.random.normal(1.9,0.4,N_t) # e as elf

x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((1*np.ones([x_h.shape[0],1]),2*np.ones([x_e.shape[0],1])))

x_te = np.concatenate((x_h_test,x_e_test))
y_te = np.concatenate((1*np.ones([N_t,1]),2*np.ones([N_t,1])))

model = Sequential()
model.add(Dense(100, input_dim=1, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])

y_tr_2 = np.empty([y_tr.shape[0],2])
y_tr_2[np.where(y_tr==1),0] = 1
y_tr_2[np.where(y_tr==1),1] = 0
y_tr_2[np.where(y_tr==2),0] = 0
y_tr_2[np.where(y_tr==2),1] = 1

history = model.fit(x_tr, y_tr_2, epochs=100, verbose=0)

plt.plot(history.history['loss'])
plt.show()

y_tr_pred = np.empty(y_tr.shape)
y_tr_pred_2 = np.squeeze(model.predict(x_tr))
for pred_ind in range(y_tr_pred_2.shape[0]):
    if y_tr_pred_2[pred_ind][0] > y_tr_pred_2[pred_ind][1]:
        y_tr_pred[pred_ind] = 1
    else:
        y_tr_pred[pred_ind] = 2

tot_correct = len(np.where(y_tr-y_tr_pred == 0)[0])
print(f'Classication accuracy (training data): {tot_correct/len(y_tr)*100}%')