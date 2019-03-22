#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:41:50 2019

@author: fredericvankelecom
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Classification trees
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import Dataset
file = 'got_explored5.xlsx'
got_explored = pd.read_excel(file)

got_data = got_explored.drop(['name',
                              'S.No',
                              'title_filled',
                              'title_filled_grouped',
                              'house_complete',
                              'age',
                              'm_mother',
                              'm_father',
                              'm_heir',
                              'dateOfBirth',
                              'culture_filled',
                              'isAlive',
                              'warrior',
                              'westerlands',
                              'lysene',
                              'crannogmen',
                              'qartheen', 
                              'asshai',
                              'first men',
                              'stormlands',
                              'tyroshi',
                              'norvoshi',
                              'qohor',
                              'valyrian',
                              'multiple_ally',
                              'myrish',
                              'sistermen',
                              'rhoynar',
                              'pentoshi',
                              'ibbenese',
                              'free folk',
                              'northern mountain clans',
                              'ghiscari',
                              'vale mountain clans',
                              'summer isles',
                              'meereen',
                              'wildlings',
                              'braavosi',
                              'riverlands',
                              'westeros',
                              'lhazareen',
                              ],
                              axis = 1)

# Creating the target variable column
got_target = got_explored.loc[:, 'isAlive']
got_target = got_target.to_frame(name=None)

###############################################################################

#0. Train-Test-Split

###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508)

###############################################################################

# 1 KNN Model

###############################################################################

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors) 
    clf.fit(X_train, y_train.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 4 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors = 3)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

cv_knn_3 = cross_val_score(knn_clf_fit, got_data, got_target, cv = 3)
print(pd.np.mean(cv_knn_3))

test_pred = knn_clf_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

# ROC_AUC score of 0.8533

###############################################################################

#2. Forward Feature Selection

###############################################################################
#These variables were obtained by a feature importance analysis

got_data2 = got_explored.loc[:,['popularity',
                             'book4_A_Feast_For_Crows',
                             'dob_before_200',
                             'out_popular',
                             'book5_A_Dance_with_Dragons',
                             'book1_A_Game_Of_Thrones',
                             'book2_A_Clash_Of_Kings',
                             'valyrian',
                             'numDeadRelations',
                             'isNoble',
                             'm_age',
                             'nightswatch',
                             'out_books',
                             'm_dateOfBirth',
                             'male_copy',
                             'm_house',
                             'no_ally',
                             'out_dead_relation',
                             'age_100',
                             'm_culture',
                             'ally_house_tully',
                             'undef_culture',
                             'lord',
                             'ally_house_lannister',
                             'ally_house_stark',
                             'isMarried',
                             'm_spouse',
                             'ally_house_tyrell',
                             'm_isAliveSpouse',
                             'ally_house_greyjoy',
                             'northmen',
                             'royals',
                             'm_isAliveFather',
                             'ally_house_baratheon',
                             'ironborn',
                             'king',
                             'm_isAliveHeir',
                             'ally_house_Martell',
                             'lady',
                             'braavosi',
                             'riverlands',
                             'wildlings',
                             'lhazareen',
                             'westeros',
                             'house_targaryen',
                             'house_tyrell']]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
            got_data2,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508)

#Choose features based on forward feature selection
feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),  
           k_features=30,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=3)

features = feature_selector.fit(np.array(X_train), y_train)  

filtered_features= X_train.columns[list(features.k_feature_idx_)]  
filtered_features

###############################################################################

#3. Building the optimal random forest using GridSearchCV

###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
            got_data2[filtered_features],
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508)

estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]


param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}

# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)

# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))

###############################################################################
# Building Random Forest Model Based on Best Parameters
###############################################################################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'gini',
                                    min_samples_leaf = 16,
                                    n_estimators = 850,
                                    warm_start = False)


c_tree_optimal_fit = rf_optimal.fit(X_train, y_train)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


cv_ctof_3 = cross_val_score(c_tree_optimal_fit, got_data, got_target, cv = 3)
print(pd.np.mean(cv_ctof_3))

rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

test_pred = c_tree_optimal_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

# ROC_AUC score of 0.8535

###############################################################################
#Check most important features of solution
###############################################################################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(20,32))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

########################
        
plot_feature_importances(rf_optimal,
                         train = X_train,
                         export = False)

print(c_tree_optimal_fit.feature_importances_)

#Get the name of important features and their contribution to the model:

importances = list(zip(rf_optimal.feature_importances_, X_train.columns))
importances.sort(reverse=True)

df_importance = pd.DataFrame(importances)

df_importance.columns = ['percent','col']

###############################################################################

#4. Logistic Regression

###############################################################################

########################
#  C = 100
########################

logreg = LogisticRegression(C = 1.0,
                            solver = 'lbfgs')


logreg_fit = logreg.fit(X_train, y_train)


logreg_pred = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

test_pred = logreg_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

cv_ctof_3 = cross_val_score(logreg_fit, got_data, got_target, cv = 3)
print(pd.np.mean(cv_ctof_3))

# ROC_AUC score of 0.8381

########################
#  C = 100
########################

logreg_100 = LogisticRegression(C = 100,
                                solver = 'lbfgs')


logreg_100_fit = logreg_100.fit(X_train, y_train)


logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))


cv_ctof_3 = cross_val_score(logreg_100_fit, got_data, got_target, cv = 3)
print(pd.np.mean(cv_ctof_3))

test_pred = logreg_100_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

# ROC_AUC score of 0.8308

########################
# C = 0.000001
########################

logreg_000001 = LogisticRegression(C = 0.000001,
                                solver = 'lbfgs')


logreg_000001_fit = logreg_000001.fit(X_train, y_train)


# Let's compare the testing score to the training score.
print('Training Score', logreg_000001_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_000001_fit.score(X_test, y_test).round(4))

cv_ctof_3 = cross_val_score(logreg_000001_fit, got_data, got_target, cv = 3)
print(pd.np.mean(cv_ctof_3))

test_pred = logreg_000001_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

# ROC_AUC score of 0.8106

###############################################################################

#5. GBM

###############################################################################

#Looking for the best model
# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}

# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)

# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)

# Fit it to the training data
gbm_grid_cv_fit = gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


###############################################################################
# Building GBM Model Based on Best Parameters
###############################################################################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.1,
                                      max_depth = 5,
                                      n_estimators = 100,
                                      random_state = 508)

gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)

gbm_optimal_score = gbm_optimal.score(X_test, y_test)

gbm_optimal_pred = gbm_optimal.predict(X_test)

# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

cv_gbm_3 = cross_val_score(gbm_optimal, got_data, got_target, cv = 3)
print(pd.np.mean(cv_gbm_3))

test_pred = gbm_optimal_fit.predict_proba(X_test)  
print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

###############################################################################

#6. Building the optimal GBM using RandomizedSearchCV

###############################################################################

X_train, X_test, y_train, y_test = train_test_split(
            got_data2,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508)

# Creating a hyperparameter grid
learn_space = pd.np.arange(0.01, 2.01, 0.05)
estimator_space = pd.np.arange(50, 1000, 50)
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'n_estimators' : estimator_space,
              'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space}



# Building the model object one more time
gbm_grid = GradientBoostingRegressor(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = RandomizedSearchCV(estimator = gbm_grid,
                                 param_distributions = param_grid,
                                 n_iter = 50,
                                 scoring = None,
                                 cv = 3,
                                 random_state = 508)

# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

###############################################################################
# Building GBM Model Based on Random Parameters
###############################################################################

gbm_optimal = GradientBoostingClassifier(criterion = 'mse',
                                      learning_rate = 0.91,
                                      max_depth = 1,
                                      n_estimators = 150,
                                      random_state = 508)

gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)

gbm_optimal_score = gbm_optimal.score(X_test, y_test)

gbm_optimal_pred = gbm_optimal.predict(X_test)

# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

cv_gbm_3 = cross_val_score(gbm_optimal, got_data, got_target, cv = 3)
print(pd.np.mean(cv_gbm_3))

test_pred = gbm_optimal_fit.predict(X_test)
test_pred_df = pd.DataFrame(data = test_pred)
test_pred_df.to_excel('test_predictions.xlsx')

model_pred = gbm_optimal_fit.predict(got_data2)
model_pred_df = pd.DataFrame(data = model_pred)
test_pred_df.to_excel('model_predictions.xlsx')

print('Accuracy on test set: {}'.format(roc_auc_score(y_test, 
                                                      test_pred [:,1])))

# ROC score of 0.8669

###############################################################################

#7. Use Confusion Matrix 

###############################################################################


# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

