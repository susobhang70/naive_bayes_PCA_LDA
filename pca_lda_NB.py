#!/usr/bin/python

import numpy as np
import csv
import sys
import math

# from sklearn.naive_bayes import GaussianNB

LDA_NUM_FEATURES_SAMPLED = 1000
TOTAL_FEATURES = 100000
KSPACE1 = 100
KSPACE2 = 500
KSPACE3 = 800
C = 1

def readfiles(type_data):
    data = []
    if type_data == 0:
        with open('dorothea_train.data', 'rb') as csvfile:
            data_lines = csv.reader(csvfile, delimiter=' ')
            for item in data_lines:
                item = np.array(item[:-1])
                data.append(item)
            csvfile.close()
    else:
        with open('dorothea_valid.data', 'rb') as csvfile:
            data_lines = csv.reader(csvfile, delimiter=' ')
            for item in data_lines:
                item = np.array(item[:-1])
                data.append(item)
            csvfile.close()
    data = np.array(data)
    return data

def readlabelfiles(returntype):
    label = []
    testlabel = []
    c1 = 0
    c2 = 0
    with open('dorothea_train.labels', 'rb') as labelfile:
        for i in labelfile:
            if int(i) == -1:
                label.append(-1)
                c1 += 1
            elif int(i) == 1:
                label.append(1)
                c2 += 1

    with open('dorothea_valid.labels', 'rb') as testlabelfile:
        for i in testlabelfile:
            if int(i) == -1:
                testlabel.append(-1)
            elif int(i) == 1:
                testlabel.append(1)
    if returntype == 0:
        return label, testlabel, c1, c2
    else:
        return label

def gaussian(mean, variance, value):
    try:
        return ( math.exp(-(math.pow(value - mean,2)/(2*variance)) ) / math.sqrt(2*math.pi*variance) )
    except:
        return 0.00000000001

def flatten(matrix):
    flat_matrix = []
    for i in matrix:
        p = i.getA1()
        flat_matrix.append(p)
    return flat_matrix

def naive_bayes(dataset, testdata, feature_count):
    label, testlabel, count_label1, count_label2 = readlabelfiles(0)

    class1 = []
    class2 = []

    for i in range(len(dataset)):
        if label[i] == -1:
            class1.append(dataset[i])
        else:
            class2.append(dataset[i])

    class1 = np.array(class1)
    class2 = np.array(class2)

    class1_mean = np.array(np.mean(class1, axis = 0))
    class2_mean = np.array(np.mean(class2, axis = 0))

    class1_cov = np.cov(class1.T)
    class2_cov = np.cov(class2.T)

    prior1 = float(count_label1) / float(len(dataset))
    prior2 = float(count_label2) / float(len(dataset))

    result_classification = []
    for i in range(len(testdata)):
        predict1 = math.log(prior1)
        predict2 = math.log(prior2)
        try:
            for j in range(len(testdata[i])):
                val = testdata[i][j]
                likelihood1 = gaussian(class1_mean[j], class1_cov[j][j], val)
                likelihood2 = gaussian(class2_mean[j], class2_cov[j][j], val)
                try:
                    predict1 += math.log(likelihood1)
                except:
                    predict1 = -100000000
                try:
                    predict2 += math.log(likelihood2)
                    if predict1 == -100000000:
                        break
                except:
                    predict1 = -100000000
                    break
        except:
            val = testdata[i]
            likelihood1 = gaussian(class1_mean, np.var(class1.T), val)
            likelihood2 = gaussian(class2_mean, np.var(class2.T), val)
            try:
                predict1 += math.log(likelihood1)
            except:
                predict1 = -100000000
            try:
                predict2 += math.log(likelihood2)
            except:
                predict1 = -100000000
        if predict1 >= predict2:
            result_classification.append(-1)
        else:
            result_classification.append(1)
    
    correct = 0
    for i in range(len(testlabel)):
        if testlabel[i] == result_classification[i]:
            correct += 1

    print float(correct)*100/float(len(testdata))

    # checkanswer = GaussianNB()
    # checkanswer.fit(dataset, label)
    # print checkanswer.score(testdata, testlabel)

def make_data_matrix_PCA():
    data_lines = readfiles(0)
    np.random.seed()
    frequency_table = np.zeros((len(data_lines), TOTAL_FEATURES), np.int)

    for i in range(len(data_lines)):
        for item in data_lines[i]:
            index = int(item) - 1
            frequency_table[i][index] = 1

    mean = np.array(np.mean(frequency_table, axis = 0))
    A = frequency_table - mean
    X = np.transpose(A)

    scatter = np.dot(X.T, X)
    pseudo_eigen_values, pseudo_eigen_vectors = np.linalg.eig(scatter)

    final_eigen_vectors = np.dot(X, pseudo_eigen_vectors)
    normalization_root = np.sqrt(pseudo_eigen_values)
    final_eigen_vectors = final_eigen_vectors/normalization_root
    indexes = pseudo_eigen_values.argsort()[::-1]

    direction_vectors1 = np.transpose(final_eigen_vectors[:, indexes[0: KSPACE1]])
    projected_data1 = np.dot(direction_vectors1, frequency_table.T)

    direction_vectors2 = np.transpose(final_eigen_vectors[:, indexes[0: KSPACE2]])
    projected_data2 = np.dot(direction_vectors2, frequency_table.T)

    direction_vectors3 = np.transpose(final_eigen_vectors[:, indexes[0: KSPACE3]])
    projected_data3 = np.dot(direction_vectors3, frequency_table.T)

    return projected_data1.T, projected_data2.T, projected_data3.T, direction_vectors1, direction_vectors2, direction_vectors3

def make_data_matrix_LDA(n_components):
    data_lines = readfiles(0)
    frequency_table = np.zeros((len(data_lines), TOTAL_FEATURES), np.int)
    label = np.array(readlabelfiles(1))
    class1 = []
    class2 = []
    X = []

    for i in range(len(data_lines)):
        for item in data_lines[i]:
            index = int(item) - 1
            frequency_table[i][index] = 1
        X.append(frequency_table[i][: LDA_NUM_FEATURES_SAMPLED])
        if label[i] == -1:
            class1.append(X[i])
        else:
            class2.append(X[i])

    X = np.array(X)

    class1 = np.array(class1)
    class2 = np.array(class2)
    class1_mean = np.array(np.mean(class1, axis = 0))
    class2_mean = np.array(np.mean(class2, axis = 0))

    A1 = class1 - class1_mean
    A2 = class2 - class2_mean

    S1 = np.dot(A1.T, A1)
    S2 = np.dot(A2.T, A2)

    SW = S1 + S2 + np.identity(len(S1[0]))

    M = np.mean(X, axis = 0)
    SB = 0

    M1 = class1_mean.reshape(LDA_NUM_FEATURES_SAMPLED,1) - M.reshape(LDA_NUM_FEATURES_SAMPLED,1)
    temp = len(class1)*np.dot(M1, M1.T)
    SB += temp

    M2 = class2_mean.reshape(LDA_NUM_FEATURES_SAMPLED,1) - M.reshape(LDA_NUM_FEATURES_SAMPLED,1)
    temp = len(class2)*np.dot(M2, M2.T)
    SB += temp

    F = np.dot(np.linalg.inv(SW), SB)
    eigen_value, eigen_vector = np.linalg.eig(F)
    indexes = eigen_value.argsort()[::-1]
    L = eigen_vector[:, indexes[0:n_components]]
    projected_data = np.dot(X, L)

    # import matplotlib.pyplot as plt
    # plt.scatter(projected_data[:,0], projected_data[:,1])
    # plt.show()

    return projected_data, L

def make_test_matrix_PCA(D1, D2, D3):
    data_lines = readfiles(1)
    frequency_table = np.zeros((len(data_lines), TOTAL_FEATURES), np.int)
    for i in range(len(data_lines)):
        for item in data_lines[i]:
            index = int(item) - 1
            frequency_table[i][index] = 1
        
    Y = frequency_table.T
    test_data1 = np.dot(D1, Y)
    test_data2 = np.dot(D2, Y)
    test_data3 = np.dot(D3, Y)
    return test_data1.T, test_data2.T, test_data3.T

def make_test_matrix_LDA(L):
    data_lines = readfiles(1)
    frequency_table = np.zeros((len(data_lines), TOTAL_FEATURES), np.int)
    X = []
    for i in range(len(data_lines)):
        for item in data_lines[i]:
            index = int(item) - 1
            frequency_table[i][index] = 1
        X.append(frequency_table[i][:LDA_NUM_FEATURES_SAMPLED])
    projected_data = np.dot(X, L)
    return projected_data

def main():
    Train1, Train2, Train3, D1, D2, D3 = make_data_matrix_PCA()
    Test1, Test2, Test3 = make_test_matrix_PCA(D1, D2, D3)
    naive_bayes(Train1, Test1, KSPACE1)
    naive_bayes(Train2, Test2, KSPACE2)
    naive_bayes(Train3, Test3, KSPACE3)
    Train_LDA, L = make_data_matrix_LDA(1)
    TestData_LDA = make_test_matrix_LDA(L)
    naive_bayes(Train_LDA, TestData_LDA, 1)
    # labels = readlabelfiles(1)
    # import matplotlib.pyplot as plt
    # C1 = [Train_LDA[i] for i in range(len(Train_LDA)) if labels[i] == -1]
    # C2 = [Train_LDA[i] for i in range(len(Train_LDA)) if labels[i] == 1]
    # plt.scatter(C1, len(C1)*[0], color ='red')
    # plt.scatter(C2, len(C2)*[0], color ='blue')
    # plt.show()

if __name__ == '__main__':
    main()