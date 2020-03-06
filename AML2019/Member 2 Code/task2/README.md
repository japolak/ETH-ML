# task2

Disease Classification

## Data

Training data consists of `X_train` of shape `(4800, 1000)` and corresponding class label with no missing values.

Since the `X` came from image data, it is assumed that the feature selection reducing the dimensions for classifier would certainly result in a loss of spatial information and thus leave to a worst prediction results. This is further confirmed by the `0.546009142434` \(k=200\) and \(k=1000\)`0.566955223235`. However, the difference is not significant and the low score reflects merely on the class imbalance.

The number of samples for class `0, 1, 2` are 600, 3600, 600 respectively.

## Effect on outliers removal

Using `LocalOutlierFactor` with default parameters, 4320 out of 4800 were chosen.

`0.676 +/- 0.022` was achieved with under-sampling, SVM combined with outliers removal. However, this resulted in a drop in the final submission score to `0.681531898726`. As in the case of SVM, lowering sample number might not aid in shaping the decision function.

## Under-sampling vs over-sampling

Scores were based on the code below 

```python
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_score = cross_val_score(
    pipeline, X, y, cv=cv, scoring='balanced_accuracy')
print("{:.3f} +/- {:.3f}".format(np.mean(cv_score), np.std(cv_score)))
```

### Over-sampling

```python
('scaler', StandardScaler()),
('smote', SMOTE()),
('clf', SVM()),
```

### Under-sampling

```python
('scaler', StandardScaler()),
('sampler', ClusterCentroids()),
('clf', SVM()),
```

### SVM

### RandomForestClassifier

Marginal return on increase on the number of estimators
