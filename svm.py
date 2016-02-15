from sklearn.svm import SVC
import numpy as np

"""
Andrew McCann
CS445 Machine Learning
Homework #3
Melanie Mitchell

THINGS TODO:
General:
    Split data to test/training (equal +/- examples)
    Find out what format SVC requires
        Format data for SVC with python

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


def test_svm():
    file_data = np.genfromtxt('spambase.data', delimiter=' ', dtype='O')

    svm = SVC()
    print(file_data[:, 0])

    # svm.fit(file_data[:, 0], file_data[:, 1:])

def cross_validate(features, labels):
    data_set = []




    return data_set

def LoadSpamData(filename="spambase.data"):
    """
    Each line in the datafile is a csv with features values, followed by a single label (0 or 1), per sample; one sample per line
    """

    unprocessed_data_file = open(filename, 'r')

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []
        split_line = line.split(',')
        for element in split_line[:-1]:
            feature_vector.append(float(element))
        features.append(feature_vector)
        labels.append(int(split_line[-1]))

    return features, labels


def BalanceDataset(features, labels):
    """
    Assumes the lists of features and labels are ordered such that all like-labelled samples are together (all the zeros come before all the ones, or vice versa)
    """

    count_0 = labels.count(0)
    count_1 = labels.count(1)
    balanced_count = min(count_0, count_1)

    # Indexing with a negative value tracks from the end of the list
    return features[:balanced_count] + features[-balanced_count:], labels[:balanced_count] + labels[-balanced_count:]


def ConvertDataToArrays(features, labels):
    """
    conversion to a numpy array is easy if you're starting with a List of lists.
    The returned array has dimensions (M,N), where M is the number of lists and N is the number of

    """

    return np.asarray(features), np.asarray(labels)


def NormalizeFeatures(features):
    """
    I'm providing this mostly as a way to demonstrate array operations using Numpy.  Incidentally it also solves a small step in the homework.
    """

    "selecting axis=0 causes the mean to be computed across each feature, for all the samples"
    means = np.mean(features, axis=0)

    variances = np.var(features, axis=0)

    "Operations in numpy performed on a 2D array and a 1D matrix will automatically broadcast correctly, if the leading dimensions match."
    features = features - means
    # features -= means

    features /= variances

    return features


def PrintDataToSvmLightFormat(features, labels, filename="svm_features.data"):
    """
    Readable format for SVM Light should be, with
    lable 0:feature0, 1:feature1, 2:feature2, etc...
    where label is -1 or 1.
    """

    if len(features) != len(labels):
        raise Exception("Number of samples and labels must match")
    dat_file = open(filename, 'w')

    for s in range(len(features)):
        if labels[s] == 0:
            line = "-1 "
        else:
            line = "1 "

        for f in range(len(features[s])):
            line += "%i:%f " % (f + 1, features[s][f])
        line += "\n"
        dat_file.write(line)
    dat_file.close()


def main():
    features, labels = LoadSpamData()
    features, labels = BalanceDataset(features, labels)
    features, labels = ConvertDataToArrays(features, labels)
    features = NormalizeFeatures(features)
    # indices = [0,1,2]
    # features = FeatureSubset(features, indices)

    PrintDataToSvmLightFormat(features, labels)


if __name__ == "__main__":
    main()
