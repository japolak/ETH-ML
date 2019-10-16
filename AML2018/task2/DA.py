#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:44:13 2018

@author: fred
"""

"""
Variable selection using random forest
@param 'x_Tr': training data, predictors
@param 'y_Tr': training data, response
@param 'nr_trees': nr of trees for random forest
@param 'class_weights': same as those by Leslie
@param 'lm': what treshhold do we set (for importance criterion)
@output: data frames with important X for train and test data [moreover list of booleans with keep= True | False]
@author: https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/ (modified by Fede)
"""
def varSelection(x_Tr, y_Tr, n_trees=1000, crit="gini", class_weights={0:8., 1:4./3, 2:8}, lm=250):
    # Train the classifier with all the variables
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=-1,criterion=crit, 
                                 class_weight=class_weights )
    clf.fit(x_Tr, y_Tr)
    # Which is the last "useful" variable?
    smallest_useful_contribution= sorted(clf.feature_importances_, reverse=True)[lm-1]
    # Columns to keep 
    keep=clf.feature_importances_ > smallest_useful_contribution
    return  keep

to_keep=varSelection(x_train_data, y_train_data)
x_relevant_train=x_train_data.iloc[:, to_keep]
x_relevant_test=x_test_data.iloc[:, to_keep]
"""
LDA. The parameters that can be tuned are 'solver' and 'shrinkage'. After a couple of attempts, this combination seemed
the best (the default solver does not allow for shrinkage and the 'eigen' seems generally worse). The shrinkage parameter
was arbitrarly tuned manually. The current value was chosen after a couple of attempts (grid search possible! 
but meaningful?)
--> in general the method seems to perform pretty well! The variable selection does not improve these results either...

QDA. I had issues with this method and always got predicted values for the most common class (1).
Looking at the probabilities for different categories I realized that many were negative infinity. 
--> I do not know why, but working with a subset of predictors ( 250- see re-written variable selection function)
), I get at least some results. No idea why. 
-- Work in progress --
"""
# create outer folds manually (sorry this is so ugly...still adjusting to python)
nr_obs=len(x_train_data)
k=10
kfolds = np.repeat(1, nr_obs/k).tolist() + \
     np.repeat(2, nr_obs/k).tolist() + \
     np.repeat(3, nr_obs/k).tolist() + \
     np.repeat(4, nr_obs/k).tolist() + \
     np.repeat(5, nr_obs/k).tolist() + \
     np.repeat(6, nr_obs/k).tolist() + \
     np.repeat(7, nr_obs/k).tolist() + \
     np.repeat(8, nr_obs/k).tolist() + \
     np.repeat(9, nr_obs/k).tolist() + \
     np.repeat(10, nr_obs/k).tolist() 
index = np.random.choice(kfolds, size=len(x_train_data), replace=False)
iter_nr = range(1, max(kfolds)+1)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def CV_LDA(nr_CV, LDA_or_QDA, x_data, y_data,ps= [1./8, 3./4, 1./8]):
    #Saving Key parameters for the methods implemented
    output= []
    confmat=[]
    for i in nr_CV:
        """
        CV strategy, the same across all methods we apply
        """
        x_train = x_data[index!=i]
        y_train = y_data[index!=i]
        
        x_test = x_data[index==i]
        y_test = y_data[index==i]
        if LDA_or_QDA=="LDA":
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.96, priors=ps)
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
            confmat.append(confusion_matrix(y_test,y_pred))
            output.append(balanced_accuracy_score(y_test,y_pred)) 
        elif LDA_or_QDA=="QDA":
            clf = QuadraticDiscriminantAnalysis(priors=ps)
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
            confmat.append(confusion_matrix(y_test,y_pred))
            output.append(balanced_accuracy_score(y_test,y_pred))     
        else:
            print "you entered the wrong method!"                    
    return output, confmat



"""
COMPARING DIFFERENT METHODS' PERFORMANCE
"""

# LDA and QDA with all predictors
res_all_LDA=CV_LDA(iter_nr, "LDA", x_train_data, y_train_data)
res_imp_LDA=CV_LDA(iter_nr, "LDA", x_relevant_train, y_train_data)
np.mean(res_all_LDA[0])-np.mean(res_imp_LDA[0])

# LDA and QDA with merely a subset of the predictors
res_all_QDA=CV_LDA(iter_nr, "QDA", x_train_data, y_train_data)
res_imp_QDA=CV_LDA(iter_nr, "QDA", x_relevant_train, y_train_data)