import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from impyute.imputation.cs import fast_knn
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% Import functions
# Import data
def importData(path, name):
    pathData = path + name + ".csv"
    dat = pd.read_csv(pathData)
    return(dat)
    
# Concatenate training and test set
def concat(train, test):
    return(train.append(test))

# Split train and test data 
def split(X, nrow_train):  
    X_train = X.iloc[0:nrow_train,:]
    X_test = X.iloc[nrow_train:,:]
    return X_train, X_test

# Missing value treatment
def imputeMV(data, strategy):
    if strategy == "mean":
        out = data.fillna(data.mean())
    elif strategy == "median":
        out = data.fillna(data.median())
    elif strategy == "mode":
        out = data.fillna(data.mode())
    return(out)
        
# Outlier detection with random forest
def outlier(X_train, y_train, cont):
    data = pd.concat([y_train, X_train], axis = 1)
    clf = IsolationForest(behaviour = "new", contamination = cont)
    clf.fit(data)
    outlier_label = clf.predict(data)
    outlier_bool = outlier_label > 0
    # Discard outliers
    out = data[outlier_bool]
    X_out = out.iloc[:, 1:]
    y_out = out.iloc[:, 0]
    return X_out, y_out

# Scaling the features
def scale(X):
    trX = StandardScaler().fit_transform(X)
    dfX = pd.DataFrame(data=trX)
    col_names = []
    for i in range(0, X.shape[1]):
        col_names.append('x' + str(i))
    dfX.columns = col_names
    return dfX

# Feature selection with random forest regressor
def selectRF(p, X_train, y_train):
    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    threshold = np.sort(importance)[::-1][p-1]
    ind = importance >= threshold
    return [X_train.iloc[:,ind], ind]

# VIF
def selectVIF(X_vif, threshold=5):
    tmp = add_constant(X_vif)
    vif = []
    for i in range(tmp.shape[1]):
        vif.append(variance_inflation_factor(tmp.values, i))
        
    vif = np.array(vif[1:])
   
    ind = vif < threshold
    return X_vif.iloc[:,ind]

# Import a pickle
import pickle
def open_pickle(file_name):
    pickle_in = open(file_name, "rb")
    file_out = pickle.load(pickle_in)
    pickle_in.close()
    return(file_out)




#%% Regression

path = "D:/ETH/A19/AdvancedML/task1/"
X_train_raw = importData(path, "X_train")
X_test_raw = importData(path, "X_test")
y_train = importData(path, "y_train")
nrow_train =  X_train_raw.shape[0] 

# Missing value treatment
X = concat(X_train_raw, X_test_raw)  
X = imputeMV(X, "median")  

X_train, X_test = split(X, nrow_train)

# Outlier detection
X_train, y_train = outlier(X_train, y_train)

# Linear regression
reg = LinearRegression()
r2 = cross_val_score(reg, X_train, y_train, scoring='r2', cv=10)
avg_r2 = np.mean(r2)

# Linear regression with RFE
reg = LinearRegression(fit_intercept = True)
rfe = RFE(reg, 820)
fit = rfe.fit(X_train, y_train)
sel = fit.support_

# Select the relevant features
X_train_sub = X_train.iloc[:, sel]
reg_sub = LinearRegression(fit_intercept = True)

r2 = cross_val_score(reg_sub, X_train_sub, y_train, scoring='r2', cv=3)
avg_r2 = np.mean(r2)


# RFE he estimator is trained on the initial set of features and the importance 
# of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute
# Then, the least important features are pruned from current set of features


# Lasso
#alpha_range = [1, 10, 100, 1000, 10000, 100000, 1000000]  #alpha =100000, r2 = 0.2322
alpha_range = [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000]
grid = dict(alpha = alpha_range)
lasso = Lasso(fit_intercept = False)
grid = GridSearchCV(lasso, grid, cv = 2, scoring =  'r2', return_train_score=False)

grid.fit(X_train, y_train)
results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

print(grid.best_score_)
print(grid.best_params_)



#%% XGBoost 

path = "D:/ETH/A19/AdvancedML/task1/"
X_train_raw = importData(path, "X_train")
X_test_raw = importData(path, "X_test")
y_train_raw = importData(path, "y_train")
nrow_train_raw =  X_train_raw.shape[0] 

X = concat(X_train_raw, X_test_raw)  
X = X.drop(["id"], axis = 1)
columns = X.columns
y_train= y_train_raw.drop(["id"], axis = 1)

#p_list = [100, 200, 300, 500, 700]
p_list = [100]
#k_list = [3, 5, 10, 20]
k_list = [5]
#cont_list = ["auto", 0.01, 0.05, 0.1]
cont_list = [0.05]
n_estimators_list = [100]
max_depth_list = [6]

results = pd.DataFrame(columns=['p','k','cont','n_estimators', 'max_depth', 'cv_r2'])
count = 0


for p in p_list:
    for k in k_list:
        for cont in cont_list:
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    X_im = fast_knn(X, k)          
                    X_im.columns = columns
                  
                    X_train, X_test = split(X_im, nrow_train_raw)
                    
                    
                    # Outlier detection
                    X_train_out, y_train_out = outlier(X_train, y_train, cont)
                    
                    
                    # Feature selection
                    outRF = selectRF(p, X_train_out, y_train_out)
                    X_train = outRF[0]
                    ind_sel = outRF[1]
                    
                    
                    # Fit xgboost
                    xgb_model = xgb.XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, objective="reg:linear", random_state=42)
                    #xgb_model.fit(X_train, y_train)
                    r2 = cross_val_score(xgb_model, X_train, y_train_out, scoring='r2', cv=5)
                    avg_r2 = np.mean(r2)
                    
                    
                    results.loc[count, 'p'] = p
                    results.loc[count, 'k'] = k
                    results.loc[count, 'cont'] = cont
                    results.loc[count, 'n_estimators'] = n_estimators
                    results.loc[count, 'max_depth'] = max_depth
                    results.loc[count, 'cv_r2'] = avg_r2
                    
                    count = count + 1


results.to_csv(path + "results_xgboost_v9.csv", index = False)


# Prediction
X_test_sel =  X_test.iloc[:,ind_sel]
xgb_fit = xgb_model.fit(X_train, y_train_out)
y_test = xgb_fit.predict(X_test_sel)  

out = pd.DataFrame()
out['id'] = range(len(X_test_raw))
out['y'] = y_test      
out.to_csv(path + '/results.csv', index=False)

#%%