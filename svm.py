from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

"""
Andrew McCann
CS445 Machine Learning
Homework #3
Melanie Mitchell

THINGS TODO:
General:
    DONE: Split data to test/training (equal +/- examples)
    DONE: Find out what format SVC requires
        DONE: Format data for SVC with python

Maybe:
    Implement methods of isolating data transformation

Experiment #1:
    Linear Kernel
    10-fold cross validation
        Used to test C values from 0 to 1 with .1 increment.
        Get largest C value from comparing accuracy (Argmax + array indeces should make trivial)
    Train a brand new SVM using this c param.
    Test on SVM model.
    Create ROC curve with 200 evenly spaced threshholds (wtf?)

Experiment #2
    Use SVM Model from #1
    Obtain weight vector w.
    From m=2 to 57
    Select features in varying number with highest weight
        Train a linear SVM on all training data using only m features and C from ex1
        Test this SVM on test data to pull accuracy

    Plot graph of accuracy vs. m

Experiment #3
    Same as two, but select the features at random, not the highest.

"""
# Number of folds
K = 10

def experiment_one(folded_array, test_set):
    """
    Experiment #1:
    Linear Kernel
    10-fold cross validation
        Used to test C values from 0 to 1 with .1 increment.
        Get largest C value from comparing accuracy (Argmax + array indeces should make trivial)
    Train a brand new SVM using this c param.
    Test on SVM model.
    Create ROC curve with 200 evenly spaced threshholds (wtf?)
    :param folded_array:
    :return:
    """

    highest_accuracy = 0
    c_max = 0
    c_vals = np.arange(.1, 1.1, 0.1)

    for j in range(len(c_vals)):
        accuracy = 0

        # Set SVM with C value
        svm = SVC(C=c_vals[j], kernel='linear')

        for i in range(K):
            array_list = list(folded_array)
            temp_test = array_list.pop(j)
            array_list = np.vstack(array_list)

            # Fit to training data
            svm.fit(array_list[:,:-1], array_list[:, -1])
            # Test
            classified = svm.predict(temp_test[:, :-1])

            accuracy += metrics.accuracy_score(temp_test[:,-1], classified)


        accuracy /= K
        if accuracy > highest_accuracy:
            print(accuracy)

            highest_accuracy = accuracy
            print("Accuracy: " + str(highest_accuracy))

            c_max = c_vals[j]
            print("C val: " + str(c_max))


    # Test this thing
    training_set = np.vstack(folded_array)
    svm = SVC(C=c_max, kernel='linear', probability=True)
    svm.fit(training_set[:, :-1], training_set[:,-1])
    predict_set = svm.predict_proba(test_set[:, :-1])

    print(predict_set)

    test_accuracy = metrics.accuracy_score(test_set[:,-1], predict_set)




def LoadSpamData(filename="spambase.data"):
    """
    Each line in the datafile is a csv with features values, followed by a single label (0 or 1), per sample; one sample per line
    """
    raw_data = np.loadtxt(filename, delimiter=',')

    # Split raw data into examples.
    negatives = raw_data[raw_data[:, -1]== 0]
    positives = raw_data[raw_data[:, -1]== 1]

    # Grab lowest count
    lowest_value = min(len(negatives), len(positives))

    # Resave the values with the extra cut off.
    negatives = negatives[:lowest_value]
    positives = positives[:lowest_value]

    # To preserve the ratio of + and - we split prior to shuffling
    mid = lowest_value//2
    negatives1 = negatives[:mid]
    negatives2 = negatives[mid:]
    positives1 = positives[:mid]
    positives2 = positives[mid:]

    positive_copy = np.copy(positives1)
    negative_copy = np.copy(negatives1)

    positive_list = np.array_split(positive_copy, K)
    negative_list = np.array_split(negative_copy, K)




    training_set = []
    for i in range(K):
        training_set.append(np.vstack((positive_list[i], negative_list[i])))
        np.random.shuffle(training_set[i])

    training_data = np.copy(training_set)
    training_data = np.vstack(training_data)

    scalar = preprocessing.StandardScaler().fit(training_data[:, :-1])

    #training_set = np.array(training_set)

    # Reform into training/test
    test_data = np.vstack((negatives2, positives2))

    # We do not want to scale the labels at the end, so skip that
    for i in range(K):
        training_set[i][:, :-1] = scalar.transform((training_set[i][:, :-1]))
    # training_set[:, :-1] = scalar.transform(training_set[:, :-1])
    # Scale test data using training data values.
    test_data[:, :-1] = scalar.transform(test_data[:, :-1])


    return training_set, test_data


def main():
    training_set, test_set = LoadSpamData()
    experiment_one(training_set, test_set)

    # features, labels = BalanceDataset(features, labels)
    # features, labels = ConvertDataToArrays(features, labels)
    # features = NormalizeFeatures(features)
    # indices = [0,1,2]
    # features = FeatureSubset(features, indices)
    # test_svm(features, labels)



if __name__ == "__main__":
    main()
