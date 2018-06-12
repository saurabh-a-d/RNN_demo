#!/usr/bin/python3

####=====================================
#### Author: Saurabh Deshpande
####=====================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time

from convert_words_to_vectors import word_to_vec, phrase_to_matrix

###================= Load train & test sets
trainf = "rnn_tutorial_reviews_cleaned_trainset.pkl"
testf = "rnn_tutorial_reviews_cleaned_testset.pkl"

trainData = []
start = time.time()
with open(trainf, "rb") as ipf:
    trainData = pickle.load(ipf)
end = time.time()
print("Loaded trainData in "+repr(int(end-start))+" sec")

testData = []
start = time.time()
with open(testf, "rb") as ipf:
    testData = pickle.load(ipf)
end = time.time()
print("Loaded testData in "+repr(int(end-start))+" sec")

print("")
dummy = input("Press ENTER to continue ...")
# Check out some details of train & test sets; display some examples
print("# training examples = "+repr(len(trainData)))
print("# testing examples = "+repr(len(testData)))

print("")
dummy = input("Press ENTER to continue ...")

idx = random.randint(0, len(trainData))
print("For sample # "+repr(idx)+" in the train data:")
print("Review:")
print(trainData[idx][2])
print("Label = "+trainData[idx][1])
print("")
dummy = input("Press ENTER to continue ...")

idx = random.randint(0, len(trainData))
print("For sample # "+repr(idx)+" in the train data:")
print("Review:")
print(trainData[idx][2])
print("Label = "+trainData[idx][1])
print("")
dummy = input("Press ENTER to continue ...")

idx = random.randint(0, len(testData))
print("For sample # "+repr(idx)+" in the test data:")
print("Review:")
print(testData[idx][2])
print("Label = "+testData[idx][1])
print("")
dummy = input("Press ENTER to continue ...")

lstNumWords = []
for sample in trainData:
    lstNumWords.append(len(sample[2].split()))
for sample in testData:
    lstNumWords.append(len(sample[2].split()))

print("Total # words in both train & test sets = "+repr(sum(lstNumWords)))
print("Average # words per review = "+repr(int(sum(lstNumWords)/len(lstNumWords))))
print("Max # words per review = "+repr(max(lstNumWords)))
print("Min # words per review = "+repr(min(lstNumWords)))

dummy = input("Press ENTER to continue ...")

###================= Calculate max sequence length
# TBD: plot histogram of numWords distribution
seqLength = 200  # Hyper-parameter
## What happens to reviews having less than seqLength words?? Last hidden-state gets carried over to time-step seqLength with "zero" inputs
## What happens to reviews having more than seqLength words?? Only 1st seqLength words considered; later words discarded
#dummy = input("Press ENTER to continue ...")
###=================

###================= Convert sequence of words to sequence of vectors in both train & test sets

sample_word, sample_vec = word_to_vec("baseball")
print(sample_word)
print(sample_vec)
print(len(sample_vec))
sample_word, sample_vec = word_to_vec("Saurabh")
print(sample_word)
print(sample_vec)
print(len(sample_vec))
print("")
dummy = input("Press ENTER to continue ...")

vecDim = len(sample_vec)
numTrainEx = len(trainData)
numTestEx = len(testData)

# each word:   vector of size vecDim
# each review: matrix of shape [seqLength, vecDim]
# train set:   tensor of shape [numTrainEx, seqLength, vecDim]
trainData_X = np.zeros([numTrainEx, seqLength, vecDim]) #numTrainEx samples of sequences of seqLength with each element of all sequences is a vector of vecDim size
# Why bunch all numTrainEx training samples together in a tensor??? Remember VECTORIZATION!!!
trainData_y = np.zeros([numTrainEx, 2]) #training labels: each label a 2-dim 1-hot vector. [1, 0] means "positive", [0, 1] means "negative"

testData_X = np.zeros([numTestEx, seqLength, vecDim]) #numTestEx samples of sequences of seqLength with each element of all sequences is a vector of vecDim size
testData_y = np.zeros([numTestEx, 2]) #testing labels (will be hidden during inference). [1, 0] means "positive", [0, 1] means "negative"

flag = input("Vectorization of training & testing sets might take a long time. Do you want to load pre-computed numpy arrays? Y/N:    ")

if flag == "Y":
    trainData_X = np.load("trainData_X.npy")
    trainData_y = np.load("trainData_y.npy")
    testData_X = np.load("testData_X.npy")
    testData_y = np.load("testData_y.npy")
else:
    print("Started vectorization of train & test datasets ...")
    start = time.time()
    for kk in range(numTrainEx):
        _, label, review = trainData[kk]
        trainData_y[kk] = [label == "positive", label == "negative"] # NOTE assigning numpy array using pure python list
        trainData_X[kk] = phrase_to_matrix(review, seqLength)[1]
        if kk == 0:
            print(trainData_y[kk])
            print(trainData_X[kk].shape)
            dummy = input("Press ENTER to continue ...")
        if kk % 50 == 0:
            print("Transformed "+repr(kk)+" training examples to matrix form ...")
    end = time.time()
    print("Vectorization of "+repr(numTrainEx)+" training examples took "+repr(int(end-start))+" sec")
    print("")
    dummy = input("Press ENTER to continue ...")
    np.save("trainData_X.npy", trainData_X)
    np.save("trainData_y.npy", trainData_y)

    start = time.time()
    for kk in range(numTestEx):
        _, label, review = testData[kk]
        testData_y[kk] = [label == "positive", label == "negative"] # NOTE assigning numpy array using pure python list
        testData_X[kk] = phrase_to_matrix(review, seqLength)[1]
        if kk == 0:
            print(testData_y[kk])
            print(testData_X[kk].shape)
            dummy = input("Press ENTER to continue ...")
        if kk % 50 == 0:
            print("Transformed "+repr(kk)+" testing examples to matrix form ...")
    end = time.time()
    print("Vectorization of " + repr(numTestEx) + " testing examples took " + repr(int(end - start)) + " sec")
    print("")
    dummy = input("Press ENTER to continue ...")
    np.save("testData_X.npy", testData_X)
    np.save("testData_y.npy", testData_y)
###=================

###================= Define the RNN, loss, avg_batch_accuracy, optimizer
# Hyperparameters
#batchSize = 20 
batchSize = 50
hiddenSize = 64
numClasses = trainData_y[0].shape[0]
print("numClasses = "+repr(numClasses))
numIters = int(numTrainEx/batchSize)

trainingBatch_X = tf.placeholder(tf.float32, shape=[None, seqLength, vecDim]) #NOTE None
trainingBatch_y = tf.placeholder(tf.float32, shape=[None, numClasses])
## See the difference between shapes of input & output data! Recall "flexibility" of the RNN architecture :-)

# Hidden layer
#cellLSTM = tf.contrib.rnn.BasicLSTMCell(hiddenSize)
cellLSTM = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize)
#dummy = input("Press ENTER to continue ...")

hiddenStatesBatch, _ = tf.nn.dynamic_rnn(cellLSTM, trainingBatch_X, dtype=tf.float32) #takes care of "unrolling" the RNN for "seqLength" steps & forward & backward pass calc.s
#NOTE only 1 layer of LSTM cells used here
print(hiddenStatesBatch.get_shape().as_list())
print("")
dummy = input("Press ENTER to continue ...")

# Gather last hiddenState for all examples in the batch
lastHiddenStateBatch = tf.gather(hiddenStatesBatch, indices=seqLength-1, axis=1) #gather all values at index (seqLength-1) on dim 1 (sizeBatch) of hiddenStateBatch
print(lastHiddenStateBatch.get_shape().as_list())
print("")
dummy = input("Press ENTER to continue ...")

# Output layer
Why = tf.Variable(tf.truncated_normal([hiddenSize, numClasses])) #Output is of shape [numClasses]
bhy = tf.Variable(tf.constant(0.1, shape=[numClasses])) #bias
print("")
dummy = input("Press ENTER to continue ...")

predictionBatch = tf.matmul(lastHiddenStateBatch, Why) + bhy  #NOTE the broadcasting in addition
print(predictionBatch.get_shape().as_list()) #probabilities of numClasses for each of batchSize examples; compare with shape of trainData_y
print("")
dummy = input("Press ENTER to continue ...")

## Where is the sigmoid layer for binary classification?? See definition of loss below

# Accuracy
comparePredictionsLabels = tf.equal(tf.argmax(predictionBatch, axis=1), tf.argmax(trainingBatch_y, axis=1)) #axis=1 ==> max along numClasses dimension: which of numClasses probabilities is max
print(comparePredictionsLabels.get_shape().as_list())
print("")
dummy = input("Press ENTER to continue ...")
avgAccuracyBatch = tf.reduce_mean(tf.cast(comparePredictionsLabels, tf.float32)) #Why cast to float32?
# No of correct predictions in the batch / total predictions made in the batch
print("")
dummy = input("Press ENTER to continue ...")

# Loss
avgLossBatch = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictionBatch, labels=trainingBatch_y)) #softmax for binary == sigmoid
# sum of losses for all predictions in the batch / total predictions made in the batch
print("")
dummy = input("Press ENTER to continue ...")

# Optimizer
train_operation_one_batch = tf.train.AdamOptimizer().minimize(avgLossBatch)
###=================

###=================  For N iterations, generate training batches & run loss minimization, plot accuracy & loss for each batch
print("Starting training ...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for kk in range(numIters):
        batch_X = trainData_X[kk*batchSize:(kk+1)*batchSize]
        batch_y = trainData_y[kk*batchSize:(kk+1)*batchSize]
        if kk == 0:
            print(batch_X.shape)
            print(batch_y.shape)
            print("")
            dummy = input("Press ENTER to continue ...")

        sess.run(train_operation_one_batch, feed_dict={trainingBatch_X: batch_X,  trainingBatch_y: batch_y}) #NOTE the diff between tf tensors & numpy arrays
        accuracyBatch = avgAccuracyBatch.eval(feed_dict={trainingBatch_X: batch_X,  trainingBatch_y: batch_y}) 
        lossBatch = avgLossBatch.eval(feed_dict={trainingBatch_X: batch_X,  trainingBatch_y: batch_y})
        # print accuracy & loss after every 10th batch
        if kk % 10 == 0:
            print("Average prediction accuracy for the batch after iteration "+repr(kk)+" = "+repr(100*accuracyBatch)+" %")

    print("Training complete ...")
    print("")
    dummy = input("Press ENTER to continue ...")

###=================  Evaluate predictions on trained model on test set (by hiding test labels)
    # Let's check predictions on 4 random examples from the test set
    labelsVec = ["positive", "negative"]
    indices = random.sample(range(testData_y.shape[0]), 4)
    predictedIdx = sess.run(tf.argmax(predictionBatch, axis=1), feed_dict={trainingBatch_X: testData_X[indices[0]].reshape((1,seqLength,vecDim))}) #batch of only 1 sample. Same tensors as used in training, since same RNN as used in training; different data fed
    #NOTE: predictionBatch depends only on tensor trainBatch_X; so, no need to feed tensor trainBatch_y
    trueIdx = np.argmax(testData_y[indices[0]])
    print("For sample "+repr(indices[0])+" in test set:")
    print("Predicted sentiment = "+labelsVec[predictedIdx[0]]+"; True sentiment = "+labelsVec[trueIdx])
    print("")
    dummy = input("Press ENTER to continue ...")

    predictedIdx = sess.run(tf.argmax(predictionBatch, axis=1), feed_dict={trainingBatch_X: testData_X[indices[1]].reshape((1,seqLength,vecDim))})
    trueIdx = np.argmax(testData_y[indices[1]])
    print("For sample "+repr(indices[1])+" in test set:")
    print("Predicted sentiment = "+labelsVec[predictedIdx[0]]+"; True sentiment = "+labelsVec[trueIdx])
    print("")
    dummy = input("Press ENTER to continue ...")

    predictedIdx = sess.run(tf.argmax(predictionBatch, axis=1), feed_dict={trainingBatch_X: testData_X[indices[2]].reshape((1,seqLength,vecDim))})
    trueIdx = np.argmax(testData_y[indices[2]])
    print("For sample "+repr(indices[2])+" in test set:")
    print("Predicted sentiment = "+labelsVec[predictedIdx[0]]+"; True sentiment = "+labelsVec[trueIdx])
    print("")
    dummy = input("Press ENTER to continue ...")

    predictedIdx = sess.run(tf.argmax(predictionBatch, axis=1), feed_dict={trainingBatch_X: testData_X[indices[3]].reshape((1,seqLength,vecDim))})
    trueIdx = np.argmax(testData_y[indices[3]])
    print("For sample "+repr(indices[3])+" in test set:")
    print("Predicted sentiment = "+labelsVec[predictedIdx[0]]+"; True sentiment = "+labelsVec[trueIdx])
    print("")
    dummy = input("Press ENTER to continue ...")

    # Calculate accuracy on entire test set
    print("Started calculating prediction results on the entire test set. WARNING: calculations might take a long time ...")
    accuracyTest = 100*avgAccuracyBatch.eval(feed_dict = {trainingBatch_X: testData_X, trainingBatch_y: testData_y}) #entire testData fed in a single batch
    print("Prediction accuracy on the test set = "+repr(accuracyTest)+" %")
