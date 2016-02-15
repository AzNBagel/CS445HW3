from sklearn.svm import SVC
import numpy as np



"""
Andrew McCann
CS445 Machine Learning
Homework #3
Melanie Mitchell

"""




def test_svm():
    file_data = np.genfromtxt('spambase.data', delimiter=',', dtype='O')

    svm = SVC()
    svm.fit(file_data[:, 0], file_data[:, 1:])


test_svm()