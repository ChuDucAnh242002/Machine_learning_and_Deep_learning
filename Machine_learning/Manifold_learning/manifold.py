import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import datasets

from pathlib import Path


# Import custom SOM implementation (location relative to home dir)
import importlib.util
#spec = importlib.util.spec_from_file_location("sklearn_som.som", "/home/kamarain/Work/ext/sklearn-som/sklearn_som/som.py")
spec = importlib.util.spec_from_file_location("sklearn_som.som", Path.home()/"Work/ext/sklearn-som-1.1.0/sklearn_som/som.py")
sklearn_som = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sklearn_som)

n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0, noise=0.1)
n_neighbors = 10
n_components = 2
X2d = np.concatenate(([X[:,0]], [X[:,2]]),axis=0).T

# Create figure
fig = plt.figure()
plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Spectral)
plt.show()

# Set for suitable plots
rnd_seed = 666
#som = sklearn_som.SOM(m=1, n=13, dim=2, random_state=rnd_seed)
som = sklearn_som.SOM(m=1, n=13, dim=2)
som.fit(X2d, shuffle=False)
#bmus = som.predict(X2d)
#X_som = som._locations[bmus,:]
X_w = som.weights

plt.scatter(X2d[:,0],X2d[:,1], c=color, cmap=plt.cm.Spectral)
plt.plot(X_w[:,0],X_w[:,1],'k-')
plt.plot(X_w[:,0],X_w[:,1],'ko')
plt.show()