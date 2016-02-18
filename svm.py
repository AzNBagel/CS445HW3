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
THRESHOLDS = 200


def experiment_one(folded_array, test_set):
    """
    Linear Kernel
    10-fold cross validation
        Used to test C values from 0 to 1 with .1 increment.
        Get largest C value from comparing accuracy (Argmax + array indeces should make trivial)
    Train a brand new SVM using this c param.
    Test on SVM model.
    Generate ROC Graph

    :param folded_array: Training set that has been split into K segments in anticipating of cross validation
    :param test_set: Stacked test test ready to use.
    :return: Returns the svm we have fitted to and the best C value we found.
    """
    highest_accuracy = 0
    c_max = 0
    c_vals = np.arange(.1, 1.1, 0.1)

    for j in range(len(c_vals)):
        accuracy = 0.0

        # Set SVM with C value
        svm = SVC(C=c_vals[j], kernel='linear')

        for i in range(K):
            array_list = list(folded_array)
            temp_test = array_list.pop(i)
            array_list = np.vstack(array_list)

            # Fit to training data
            svm.fit(array_list[:, :-1], array_list[:, -1])
            # Test
            classified = svm.predict(temp_test[:, :-1])
            accuracy += metrics.accuracy_score(temp_test[:, -1], classified)

        accuracy /= K
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            c_max = c_vals[j]

    # Test this thing
    training_set = np.vstack(folded_array)
    svm = SVC(C=c_max, kernel='linear', probability=True)
    svm.fit(training_set[:, :-1], training_set[:, -1])
    predict_set = svm.predict_proba(test_set[:, :-1])
    predict_set = predict_set[:, 1]

    # Make a copy for calculations used later.
    saved_results = np.copy(predict_set)

    predict_set[predict_set >= .5] = 1
    predict_set[predict_set < .5] = 0

    test_accuracy = metrics.accuracy_score(test_set[:, -1], predict_set)
    test_recall = metrics.recall_score(test_set[:, -1], predict_set)
    test_precision = metrics.precision_score(test_set[:, -1], predict_set)

    print("Accuracy: " + str(test_accuracy))
    print("Recall: " + str(test_recall))
    print("Precision: " + str(test_precision))

    # Generation of the ROC Curve
    true_pos = []
    false_pos = []
    # Create some threshold values.
    threshold_array = []
    for i in range(THRESHOLDS):
        threshold_array.append((1. * i) / THRESHOLDS)

    # Need to evalute different threshold levels
    # Against the saved_result(threshold_set) and labels
    labels = test_set[:, -1]
    for threshold in threshold_array:
        # Make a new copy of saved_set so we can change values
        threshold_set = np.copy(saved_results)

        # Change values based on threshold
        threshold_set[threshold_set < threshold] = 0.0
        threshold_set[threshold_set >= threshold] = 1.0

        # Need to gen TPR = TP/(TP+FN)
        #             FPR = FP/(FP+TN)
        # Metrics has a recall score
        true_pos.append(metrics.recall_score(labels, threshold_set))
        # Metrics ROC Curve method doesn't seem to have a way to adjust threshold, or I'd
        # Just use that to rip the FPR out.
        fp = 0.0
        tn = 0.0
        for i in range(len(threshold_set)):

            if threshold_set[i] == 1.0 and labels[i] == 0.0:
                fp += 1.0
            if threshold_set[i] == 0.0 and labels[i] == 0.0:
                tn += 1.0
        false_pos.append(fp / (fp + tn))

    false_pos.append(0)
    true_pos.append(0)
    print(false_pos[::-1])
    print(true_pos[::-1])

    print("Generating ROC Curve")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(false_pos[::-1], true_pos[::-1])
    plt.axis([0, 1, 0, 1])
    plt.show()

    # return the svm so that we can use it in other sets.
    return svm, c_max


def experiment_two(svm, c_max, training_set, test_set):
    """
    Use SVM Model from #1
    Obtain weight vector w.
    From m=2 to 57
    Select features in varying number with highest weight
        Train a linear SVM on all training data using only m features and C from ex1
        Test this SVM on test data to pull accuracy

    :param svm:
    :param c_max:
    :param training_set:
    :param test_set:
    :return:
    """
    weights = svm.coef_
    training_set = np.vstack(training_set)

    weight_vector = weights[0]
    weight_vector = np.absolute(weight_vector)
    # Numpy's amazing methods are at work here to return the indices
    index_array = np.argsort(weight_vector)
    # Sorted from min to max, so need to flip it.
    index_array = np.flipud(index_array)

    labels = training_set[:, -1]
    accuracy_list = []
    m_list = []
    m_list.append(index_array[0])
    for i in range(1, 57):
        m_list.append(index_array[i])
        if i == 4:
            print("Top 5:" + str(m_list))
        # Following format of Jordan's function for pulling indices
        features = training_set[:, m_list]
        test_features = test_set[:, m_list]
        svm_two = SVC(C=c_max, kernel='linear')
        svm_two.fit(features, labels)
        results = svm_two.predict(test_features)
        results[results >= .5] = 1
        results[results < .5] = 0

        accuracy_list.append(metrics.accuracy_score(test_set[:, -1], results))

    # Plot graph
    accuracy_list.insert(0, 0)
    index = np.arange(0, 57)
    print("Generating graph for largest magnitude features")
    plt.title("Largest Magnitude Feature Selection")
    plt.xlabel("# of features")
    plt.ylabel("Accuracy of features")
    plt.plot(index, accuracy_list)
    plt.axis([0, 57, 0, 1])
    plt.show()


def experiment_three(svm, c_max, training_set, test_set):
    """

    :param svm:
    :param c_max:
    :param training_set:
    :param test_set:
    :return:
    """
    weights = svm.coef_
    training_set = np.vstack(training_set)

    weight_vector = weights[0]
    weight_vector = np.absolute(weight_vector)
    # Numpy's amazing methods are at work here to return the indices
    index_array = np.argsort(weight_vector)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Create a random permutation per Experment 3 guidelines.
    index_array = np.random.permutation(index_array)

    labels = training_set[:, -1]
    accuracy_list = []
    m_list = []
    m_list.append(index_array[0])

    for i in range(1, 57):
        m_list.append(index_array[i])

        # Following format of Jordan's function for pulling indices
        features = training_set[:, m_list]
        test_features = test_set[:, m_list]
        svm_two = SVC(C=c_max, kernel='linear')
        svm_two.fit(features, labels)
        results = svm_two.predict(test_features)
        results[results >= .5] = 1
        results[results < .5] = 0

        accuracy_list.append(metrics.accuracy_score(test_set[:, -1], results))

    accuracy_list.insert(0, 0)

    index = np.arange(0, 57)

    print("Generating graph for Random Features")
    plt.title("Random Feature Selection Results")
    plt.xlabel("# of features ")
    plt.ylabel("Accuracy of features")
    plt.plot(index, accuracy_list)
    plt.axis([0, 57, 0, 1])
    plt.show()


def load_spam_data(filename="spambase.data"):
    """

    :param filename:
    :return:
    """
    raw_data = np.loadtxt(filename, delimiter=',')

    # Split raw data into examples.
    negatives = raw_data[raw_data[:, -1] == 0]
    positives = raw_data[raw_data[:, -1] == 1]

    # Grab lowest count
    lowest_value = min(len(negatives), len(positives))
    # Get a number we can split in ten.
    lowest_value -= (lowest_value % 10)

    # Resave the values with the extra cut off.
    negatives = negatives[:lowest_value]
    positives = positives[:lowest_value]

    # To preserve the ratio of + and - we split prior to shuffling
    mid = lowest_value // 2
    negatives1 = negatives[:mid]
    negatives2 = negatives[mid:]
    positives1 = positives[:mid]
    positives2 = positives[mid:]

    test_data = np.vstack((positives2, negatives2))
    training_data = np.vstack((positives1, negatives1))

    np.random.shuffle(training_data)

    # pulled from my HW2
    scalar = preprocessing.StandardScaler().fit(training_data[:, :-1])

    training_data[:, :-1] = scalar.transform(training_data[:, :-1])
    test_data[:, :-1] = scalar.transform(test_data[:, :-1])

    training_set = np.array_split(training_data, K)

    return training_set, test_data


def main():
    print("Loading data...")
    training_set, test_set = load_spam_data()
    print("Running Experiment 1")
    svm, c_val = experiment_one(training_set, test_set)
    print("Running Experiment 2")
    experiment_two(svm, c_val, training_set, test_set)
    print("Running Experiment 3")
    experiment_three(svm, c_val, training_set, test_set)
    print("SVM Experiments concluded.")


if __name__ == "__main__":
    main()
