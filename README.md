# Naive Bayes and PCA/LDA

Python implementation of Naive Bayes and dimension reduction techniques - PCA & LDA, and running them on different datasets

## Naive Bayes Classifier

Run: `naive_bayes.py`

This automatically runs on the census dataset:  
- `census-income.data`
- `census-income.names`
- `census-income.test`

The full versions of these files can be downloaded from <https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29> and put in this directory.

- `naive_bayes.py` - This implements the classifier and runs on the cenus-income dataset mentioned above.

- **Log probabilities** were used to avoid calculation errors due to very small numbers. A small logconst was added to the probabilities and then the log was taken. These log probabilities were then summed up.

- While training, if a missing entry occurs, we can do two things, either ignore the sample or take only the values of the features which aren’t missing. In this case I chose to take the values and not ignore the sample per se - sort of treating the `?` or missing values as an attribute. While testing, we just use the features which have no missing values and ignore the ones with a missing value.

## PCA & LDA

This work was done on the Dorothea dataset <https://archive.ics.uci.edu/ml/datasets/Dorothea>
Training was done on the `train` file and testing on the `valid` file.

Run: `pca_lda_NB.py`

This implements the PCA technique.

Since the original dataset consists of a really large number of features - 100000 features, there are two ways one can go about the task - use the so called ‘kernel trick’ for calculating the eigen vectors of ATA matrix or to select a random subsample of the 100000 features and then implement a PCA on that feature set. I've gone with the former approach. More about it here:
<http://stats.stackexchange.com/questions/7111/how-to-perform-pca-for-data-of-very-high-dimensionality>
<http://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca>
Here we have separate train and test data. For training, all the 600 samples were used. For testing, 200 of the 350 samples were used.

- `TOTAL_FEATURES`: Total number of features in the dataset

- `KSPACE1`, `KSPACE2`, and `KSPACE3`: There are the three K's for three k-dimensional PCA space reductions. As said above, I used the kernel trick as the number of features is too high. So since the AAT matrix has only max of 800 eigen values in this case, the highest K that I could use here (`KSPACE3`) is 800.

- `LDA_NUM_FEATURES_SAMPLED`: Features sampled from the dataset for LDA reduction