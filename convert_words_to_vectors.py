#! /usr/bin/python3

import os
import numpy as np

vocab = np.load('vocabulary.npy')
wordsList = vocab.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('vocabVectors.npy')

def word_to_vec(word):
    try:
        Index = wordsList.index(word)
        return word, wordVectors[Index]
    except ValueError:
        return wordsList[-1], wordVectors[-1] #UNK

def phrase_to_matrix(phrase, seqL):
    # Consider only 1st seqL words in the phrase
    _, sampleVec = word_to_vec("baseball")
    retMatrix = np.zeros([seqL, len(sampleVec)]) #word-vectors stacked row-wise
    retPhrase = []
    counter = 0
    for word in phrase.split()[:seqL]:
        retWord, retVec = word_to_vec(word)
        retPhrase.append(retWord)
        retMatrix[counter] = retVec
        counter += 1

    return " ".join(retPhrase), retMatrix
"""
retPhrase, matPhrase = phrase_to_matrix("baseball is sport", 3)
print(retPhrase)
print(matPhrase.shape)
print(matPhrase)

temp = np.zeros([2, 3, 50])
temp[0] = matPhrase
print(temp)

retPhrase, matPhrase = phrase_to_matrix("what good is charity for", 4)
print(retPhrase)
print(matPhrase.shape)
print(matPhrase)
"""
