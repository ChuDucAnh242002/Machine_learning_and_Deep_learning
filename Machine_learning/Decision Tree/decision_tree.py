import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import export_text

file = "data/PlayTennis.dat"

X = []
y = []

with open(file, "r", encoding='ISO-8859-1') as f:
    for line in f:
        
        # Skip first line
        if line.startswith("Day"):
            print(line)
            parts = line.strip().split(",")
            feat_names = parts[1:5]
            continue
        
        parts = line.strip().split(",")
        
        outlook = parts[1]
        temperature = parts[2]
        humidity = parts[3]
        wind = parts[4]
        playtennis = parts[5]
        
        X.append([outlook, temperature, humidity, wind])
        y.append(playtennis)
        
X = np.array(X)
y = np.array(y)
print(X)
print(y)