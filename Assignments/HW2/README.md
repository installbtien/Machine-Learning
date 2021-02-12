# HW2: Binary Classification
## Task: Binary Classification
whether the income of an individual exceeds $50000 or not?

## Dataset
https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)

* remove unnecessary attributes
* balance the ratio between positively and negatively labeled data.

## Feature Format
* **train.csv, test_no_label.csv**
  + text-based raw data
  + unnecessary attributes removed, positive/negative ratio balanced.
  
* **X_train, Y_train, X_test**
  + discrete features in train.csv => one-hot encoding in X_train (education, martial state...)
  + continuous features in train.csv => remain the same in X_train (age, capital losses...).
  + X_train, X_test : each row contains one 510-dim feature represents a sample.
  + Y_train: label = 0 means  “<= 50K” 、 label = 1 means  “ >50K ”
