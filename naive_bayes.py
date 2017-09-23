#!/usr/bin/python

import sys
import math
from collections import Counter

type_of_data = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
                0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,\
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

distinct_v = [91, 9, 52, 47, 17, 1240, 3, 7, 24, 15, 5,\
              10, 2, 3, 6, 8, 132, 113, 1478, 6, 6, 51,\
              38, 8, 0, 10, 9, 10, 3, 4, 7, 5, 43, 43,\
              43, 5, 3, 3, 3, 53, 2, 2]

def main():

    # Opening the file, and extracting each line
    fp1 = open('./census-income.data', 'r')
    fp2 = open('./census-income.test', 'r')
    train_data = fp1.readlines()
    test_data  = fp2.readlines()
    data_lines = train_data[:]
    test_lines = test_data[:]

    # The first line consists of all features, we extract number and names.
    feature_names = data_lines[0].split(", ")
    feature_count = len(feature_names)
    instance_count = len(data_lines)
    test_count = len(test_lines)
    discrete_distribution = []

    # Initializing some values
    count_label1 = 0
    count_label2 = 0
    label = [0 for x in range(instance_count)]
    F = [0.00 for x in range(feature_count)]

    # Classifying each instance using the labels
    for i in range(instance_count):
        items = data_lines[i].split(", ")
        items[feature_count - 1] = "".join(items[feature_count - 1].split()).rstrip('.')
        if items[feature_count - 1] == "-50000":
            label[i] = 1
            count_label1 += 1
        else:
            label[i] = 2
            count_label2 += 1
    # print count_label1, count_label2

    # Finding frequencies of data
    for j in range(feature_count - 1):
        temp = []
        temp1 = []
        temp2 = []
        for i in range(instance_count):
            items = data_lines[i].split(", ")
            val = items[j].strip()
            if type_of_data[j] == 0:
                if label[i] == 1:
                    temp1.append(val)
                else:
                    temp2.append(val)
        if type_of_data[j] == 0:
            counter1 = Counter(temp1)
            counter2 = Counter(temp2)
            tempdict1 = {}
            tempdict2 = {}
            for key, value in counter1.iteritems():
                tempdict1[key] = float(value + 1)/float(count_label1 + distinct_v[j])
            for key, value in counter2.iteritems():
                tempdict2[key] = float(value + 1)/float(count_label2 + distinct_v[j])
            
            temp.append(tempdict1)
            temp.append(tempdict2)
        
        discrete_distribution.append(temp)

    total = count_label1 + count_label2
    prior1 = float(count_label1)/float(total)
    prior2 = float(count_label2)/float(total)

    # cross validation test
    count_correct = 0
    for i in range(test_count):
        probability1 = math.log(prior1)
        probability2 = math.log(prior2)
        items = test_lines[i].split(", ")
        items[feature_count - 1] = "".join(items[feature_count - 1].split()).rstrip('.')
        for j in range(feature_count - 1):
            temp1 = 0
            temp2 = 0
            val = items[j].strip()
            if type_of_data[j] == 0:
                try:
                    temp1 = discrete_distribution[j][0][val]
                except:
                    temp1 = float(1)/float(count_label1 + distinct_v[j])
                try:
                    temp2 = discrete_distribution[j][1][val]
                except:
                    temp2 = float(1)/float(count_label2 + distinct_v[j])
                probability1 += math.log(temp1)
                probability2 += math.log(temp2)
        if (probability1 > probability2 and items[feature_count - 1] == "-50000") or \
            (probability2 > probability1 and items[feature_count - 1] == "50000+"):
            count_correct += 1

    print count_correct, test_count
    print float(count_correct)*100/float(test_count)

if __name__ == '__main__':
    main()