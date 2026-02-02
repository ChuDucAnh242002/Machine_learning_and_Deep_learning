import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')

file = open(vocabulary_file, 'r', encoding="utf8")
words = [x.rstrip().split(' ')[0] for x in file.readlines()]

# Read word vectors
print('Read word vectors...')
file = open(vocabulary_file, 'r', encoding="utf8")
vectors = {}
for line in file:
    vals = line.rstrip().split(' ')
    vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)} # vocab is the word first and then the index later
ivocab = {idx: w for idx, w in enumerate(words)} # inverse vocab is the index first and then the word later

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

def find_vector(word_x, word_y, word_z):
    word_x_idx = vocab[word_x]
    word_y_idx = vocab[word_y]
    word_z_idx = vocab[word_z]

    vector_x = W[word_x_idx]
    vector_y = W[word_y_idx]
    vector_z = W[word_z_idx]

    vector_z2 = vector_z + (vector_y - vector_x)
    return vector_z2

def find_similar_words(vector_input, words_idx, word_num):
    distances = []
    distances = np.linalg.norm(vector_input - W, axis = 1)

    similar_word_index = np.argsort(distances)
    similar_words = {}
    for index in similar_word_index:
        if index not in words_idx:
            similar_words[index] = distances[index]
        if len(similar_words) == word_num: break
    return similar_words

# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")

    if input_term == 'EXIT':
        break
    words = input_term.rstrip().split("-")
    if len(words) != 3:
        print("Input needs 3 words")
        continue
    word_x = words[0]
    word_y = words[1]
    word_z = words[2]
    words_idx = [vocab[word_x], vocab[word_y], vocab[word_z]]

    vector_z2 = find_vector(word_x, word_y, word_z)
    
    similar_words = find_similar_words(vector_z2, words_idx, 2)

    print("\n                               Word       Distance\n")
    print("---------------------------------------------------------\n")
    for index in similar_words:
        print("%35s\t\t%f\n" % (ivocab[index], similar_words[index]))