#%% Utils file
import os
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import itertools

# Make Path To File
def makePath():
    path = os.getcwd()
    print("Check current path mathches file directory:")
    print(path)
    return(path)

# Import Data
def importData(name, path=os.getcwd()):
    pathData = path + "/" + name + ".csv"
    data = pd.read_csv(pathData)
    print(name, "imported!")
    return (data)

# Concatenate / Unconcatenate training and test set
#   n1 = train obs , n2 = test obs ( use of global variables! )
#   concat = merge , unconcat = split
def concat(train, test, labels=None):
    global n1, n2, n, m
    n1 = train.shape[0]
    n2 = test.shape[0]
    m = test.shape[1]
    n = n1 + n2
    out = train.append(test)
    if labels is None:
        return out
    else:
        return out, labels

def unconcat(data):
    if (n == data.shape[0]) and (n1+n2 == n):
        train = data.iloc[0:(n-n2), :]
        test = data.iloc[(n-n2):, :]
    else:
        print('Dimensions dont match! (You lost some observaitions)')
    return train, test

# Remove Useless Columns
def removeId(data):
    out = data.drop(['id'], axis=1)
    return out

def removeConstants(data, add_intercept=False):
    tmp = data.fillna(data.median())
    out = data.loc[:, (tmp != tmp.iloc[0]).any()]
    if add_intercept:
        out = out.insert(0, "intercept", 1)
    return out

def removeZeros(data, threshold=0.2):
    counts = np.invert(data.astype(bool)).sum(axis=0)
    print("Zero values: ", round(counts.sum()*100/(n*m), 2),
          "% (", counts.sum(), "/", n*m, ")")
    for i in range(counts.size):
        if counts[i] > 0:
            print("Column", counts.index[i], ": ", round(
                counts[i]*100/n, 2), "% (", counts[i], "/", n, ")")
    if threshold == 0:
        out = data.loc[:, (counts == 0)]
    else:
        out = data.loc[:, (counts < threshold*n)]
    out = out.replace(to_replace=0, value=np.nan)
    print("Columns with >", threshold*100,
          "% zero values removed! Other zeros = NaN!")
    return out


# Missing Values
def missingValues(data, opt='check', **kwargs):
    # Check for missing values
    if opt == 'check':
        missing = data.isna().sum().sum()
        print("Missing values: ", round(missing*100/(n*m), 2),
              "%  (", missing, "/", n*m, ")")
    else:
        # Choose method
        if opt == 'mean':
            out = data.fillna(data.mean())
        elif opt == 'median':
            out = data.fillna(data.median())
        else:
            # Iterative imputers
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            if opt == 'bayesian':
                from sklearn.linear_model import BayesianRidge
                estim = BayesianRidge(n_iter=100, **kwargs)
            elif opt == 'extra':
                from sklearn.ensemble import ExtraTreesRegressor
                estim = ExtraTreesRegressor(n_estimators=50, max_features=0.5, min_impurity_decrease=1e-3,
                                            min_samples_split=5, min_samples_leaf=2, n_jobs=-1, **kwargs)
            elif opt == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                estim = KNeighborsRegressor(n_jobs=-1, **kwargs)
            imp = IterativeImputer(
                estimator=estim, max_iter=5, n_nearest_features=100, verbose=2, random_state=0)
            out = pd.DataFrame(imp.fit_transform(
                data), columns=data.columns, index=data.index)
        print("Missing values imputed using", opt, "method!")
        return out


# Selecting Not Collinear Features
def selectVIF(data, threshold=5):
    vif = np.linalg.inv(data.corr().values).diagonal()
    ind = vif < threshold
    out = data.iloc[:, ind]
    print("Collinear features: ", round(np.invert(ind).sum()*100/m, 2),
          "%  (", np.invert(ind).sum(), "/", m, ")")
    print("Collinear features with VIF >", threshold, "removed!")
    return out


# Removing Outliers
def removeOutliers(train, labels=None, opt='isolation', cont='auto', rerun=100, outlier_importance=20, max_features=0.2, max_samples=0.2, random_state=0, **kwargs):
    # Set seed and data size
    n1, m = train.shape
    np.random.seed(random_state)

    # Merge into one dataset with labels
    if labels is None: data = train
    else: data = pd.concat([train, labels], axis=1)

    # Define functions for interation of estimators
    def IterateResults(estimator, data, rerun):
        score = np.zeros(n1)
        print("Outlier detection: Iterating", opt, "estimator", rerun, "times.")
        print("Cummulative outliers found")

        def resample_score(seed):
            np.random.seed(seed)
            return estimator.fit(data).decision_function(data)

        mapping = map(resample_score, range(random_state, random_state+rerun))

        for i in mapping:
            # Give more weights to outliers found
            i[i < 0] = i[i < 0]*outlier_importance
            score += i                                 
            print((score < 0).sum(), end="->")
        print("Done!")
        return score/rerun

    def MahalanobisDist(data):
        
        def is_pos_def(A):
            if np.allclose(A, A.T):
                try: np.linalg.cholesky(A) ; return True
                except np.linalg.LinAlgError: return False
            else: return False

        covar = np.cov(data, rowvar=False)
        if is_pos_def(covar):
            covar_inv = np.linalg.inv(covar)
            if is_pos_def(covar_inv):
                mean = np.mean(data,axis=0)
                diff = data - mean
                md = np.sqrt(diff.dot(covar_inv).dot(diff.T).diagonal())
                return md
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")


    # Choose method
    if opt == 'isolation':
        from sklearn.ensemble import IsolationForest
        estim = IsolationForest(contamination=cont, behaviour='new', max_samples=max_samples,
                                max_features=max_features, n_estimators=50, n_jobs=-1, **kwargs)
        decision = estim.fit(data).predict(data)
        if (rerun > 0):
            decision = IterateResults(estim, data, rerun)

    if opt == 'lof':
        from sklearn.neighbors import LocalOutlierFactor
        estim = LocalOutlierFactor(contamination=cont, n_neighbors=55, n_jobs=-1)
        decision = estim.fit_predict(data)

    if opt == 'svm':
        from sklearn.svm import OneClassSVM
        if cont == 'auto':
            cont = 0.01
        estim = OneClassSVM(nu=cont, gamma='scale', tol=1e-3)
        decision = estim.fit(data).predict(data)

    if opt == 'covariance':
        if cont == 'auto': cont = 3
        MD = MahalanobisDist(data.values)
        std = np.std(MD)
        mean = np.mean(MD)
        k = 3. * std if True else 2. * std
        high, low = mean + k, mean - k
        decision = (MD >= high)*(-2) + (MD <= low)*(-2) + 1

    # Print summary information
    index = decision < 0
    print("Outlier values: ", round(index.sum()*100/n1, 3),
          "%  (", index.sum(), "/", n1, ")")
    print("Outlier values indecies:")
    for i in data[index].index:
        print(i, end=' ')
    print()
    if index.sum()/n1 > 0.1:
        print("Warning! More than 10% of training observations deleted!")
    # Discard outliers
    out = data[np.invert(index)]
    if labels is None:
        return out
    else:
        train = out.iloc[:, 0:m]
        labels = pd.DataFrame(out.iloc[:, -1])
        return (train, labels)


# Scaling Features
def scaleFeatures(data, opt='standard', **kwargs):
    if opt == 'standard':
        scl = sklearn.preprocessing.StandardScaler(**kwargs)
    elif opt == 'robust':
        scl = sklearn.preprocessing.RobustScaler(**kwargs)
    elif opt == 'minmax':
        scl = sklearn.preprocessing.MinMaxScaler(**kwargs)
    elif opt == 'norm':
        scl = sklearn.preprocessing.Normalizer(**kwargs)
    elif opt == 'gaussian':  # doesn't work! no idea why
        scl = sklearn.preprocessing.PowerTransformer(method='yeo-johnson')
    elif opt == 'quantile':
        scl = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal')
    out = pd.DataFrame(scl.fit_transform(data), columns=data.columns)
    print("Features scaled using", opt, "scaling method!")
    return out

# Feature Selection
def selectFeatures(X1, X2, y1, opt='covariance', features=False, threshold = None, setting = 'regression'):
    n,p = X1.shape
    train = X1.values
    test = X2.values
    labels = y1.values.ravel()

    if opt == 'lasso':
        def scoreLasso(train,labels):
            from sklearn.linear_model import LassoCV
            estim = LassoCV(cv = 10, random_state=1, n_jobs=-1)
            estim.fit(train, labels)
            return abs(estim.coef_)
        scores = scoreLasso(train,labels)
        t = 0.001
    
    if opt == 'covariance':
        cov = np.cov(train, labels, rowvar=False)
        scores = abs(cov[p, 0:p])
        t = 0.1
        
    if opt == 'randomforest':
        def scoreRF(train,labels):
            from sklearn.ensemble import RandomForestRegressor
            estim = RandomForestRegressor(
                n_estimators=500, n_jobs=-1, random_state=0)
            estim.fit(train, labels)
            return estim.feature_importances_
        scores = scoreRF(train,labels)
    

    if opt == 'Fscore':
        if setting == 'regression':
            from sklearn.feature_selection import f_regression
            scores , _ = f_regression(train,labels)
        if setting == 'classification':
            from sklearn.feature_selection import f_classif
            scores, _ = f_classif(train, labels)

    if opt == 'mutualinfo':
        if setting == 'regression':
            from sklearn.feature_selection import mutual_info_regression
            scores = mutual_info_regression(train, labels)
        if setting == 'classification':
            from sklearn.feature_selection import mutual_info_classif  
            scores = mutual_info_classif(train,labels)
    
    if features == False:
        if threshold is None : 
            if opt == 'lasso': threshold = t
            if opt == 'covariance': threshold = t
            else: threshold = np.mean(scores)
        mask = scores > threshold

    else:
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores, kind="mergesort")[-features:]] = 1

    trainf = X1.iloc[:, mask]
    testf = X2.iloc[:, mask]
    print("Feature selection using",opt,"method")
    print("Best",mask.sum(),"features selected!")
    return trainf, testf

def compressFeatures(train, labels):
    return train
    

def outputData(train, labels, test):
    X_train = pd.DataFrame(train)
    X_train['id'] = range(train.shape[0])
    X_train.to_csv(path + '/X_train_jakub.csv', index=False)

    X_test = pd.DataFrame(test)
    X_test['id'] = range(test.shape[0])
    X_test.to_csv(path + '/X_test_jakub.csv', index=False)

    y_train = pd.DataFrame(labels)
    y_train['id'] = range(labels.shape[0])
    y_train.to_csv(path + '/y_train_jakub.csv', index=False)

def submissionFile(predictions):
    out = pd.DataFrame()
    out['id'] = range(len(predictions))
    out['y'] = predictions
    out.to_csv(path + '/results.csv', index=False)

