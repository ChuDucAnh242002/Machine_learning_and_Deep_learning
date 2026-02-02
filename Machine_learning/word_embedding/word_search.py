# import random
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

def find_similar_words(input_term, word_num):
    input_term_idx = vocab[input_term]
    vector_input = W[input_term_idx]
    distances = []
    distances = np.linalg.norm(vector_input - W, axis = 1)

    similar_word_index = np.argsort(distances)[:word_num]

    # similar_words = {}
    # for index in similar_word_index:
    #     similar_words[index] = distances[index]
    # return similar_words

    similar_words = np.zeros((len(similar_word_index), 2))
    
    # Populate the array with index and distance
    for i, index in enumerate(similar_word_index):
        similar_words[i] = [index, distances[index]]
    
    return similar_words
    
# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        similar_words = find_similar_words(input_term, 3)
        print(similar_words.shape)

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for index in similar_words:
            print("%35s\t\t%f\n" % (ivocab[index[0]], index[1]))