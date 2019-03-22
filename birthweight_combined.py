# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:39:08 2019

@author: Frederic Van Kelecom 

Purpose: 
This program seeks to determine the best model to determine the weight of 
newborn babies using the dataset provided in class. It will test the accuracy 
of different models and feature combinations using mainly R^2. 

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO 
from IPython.display import Image 
import pydotplus 

from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor 
import statsmodels.formula.api as smf 
from sklearn.metrics import mean_squared_error, r2_score


pd.set_option('display.max_columns',100)


file = 'birthweight_feature_set.xlsx'
birthweight = pd.read_excel(file)


##############################################################################

# Fundamental Dataset Exploration

##############################################################################

# Column names
birthweight.columns


# Displaying the first rows of the DataFrame
print(birthweight.head())


# Dimensions of the DataFrame
birthweight.shape


# Information about each variable
birthweight.info()


# Descriptive statistics
birthweight.describe().round(2)


birthweight.sort_values('bwght', ascending = False)


###############################################################################

# Imputing Missing Values

###############################################################################

print(
      birthweight
      .isnull()
      .sum()
      )



for col in birthweight:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(int)

# Let's first explore the columns with missing values


# Creating a dropped dataset and graphing it
df_dropped = birthweight.dropna()

sns.distplot(df_dropped['meduc'])
plt.show()
sns.distplot(df_dropped['npvis'])
plt.show()
sns.distplot(df_dropped['feduc'])
plt.show()


# Everything else being filled with the median

for col in birthweight:

    """ Filling the missing values with the median """
    
    if birthweight[col].isnull().any():
        fill = df_dropped[col].median()
        birthweight[col] = birthweight[col].fillna(fill)

# Checking the overall dataset to see if there are any remaining
# missing values
        
print(
      birthweight
      .isnull()
      .any()
      .any()
      )

###############################################################################

# Outlier Analysis

###############################################################################


birthweight_quantiles = birthweight.loc[:, :].quantile([0.05,
                                                        0.40,
                                                        0.60,
                                                        0.80,
                                                        0.95])

print(birthweight_quantiles)
    



###############################################################################

# Visual EDA (Histograms)

###############################################################################

#Characteristics of the mother

plt.subplot(2, 2, 1)
sns.distplot(birthweight['mage'],
             bins = 35,
             color = 'g')

plt.xlabel("Mother's age")


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['meduc'],
             bins = 30,
             color = 'y')

plt.xlabel("Mother's education")



########################


plt.subplot(2, 2, 3)
plt.hist(birthweight['mwhte'],
             color = 'orange')

plt.xlabel('Mother white')



########################


plt.subplot(2, 2, 4)

plt.hist(birthweight['mblck'],
             color = 'r')

plt.xlabel('Mother Black')



plt.tight_layout()
plt.savefig('Birthweight Mother.png')

plt.show()



########################

#Characteristics of the father

plt.subplot(2, 2, 1)
sns.distplot(birthweight['fage'],
             bins = 35,
             color = 'g')

plt.xlabel("Father's age")


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['feduc'],
             bins = 30,
             color = 'y')

plt.xlabel("Father's education")



########################


plt.subplot(2, 2, 3)
plt.hist(birthweight['fwhte'],
             color = 'orange')

plt.xlabel('Father white')



########################


plt.subplot(2, 2, 4)

plt.hist(birthweight['fblck'],
             color = 'r')

plt.xlabel('Father Black')



plt.tight_layout()
plt.savefig('Birthweight Father.png')

plt.show()



########################

#Rest of the information

plt.subplot(2, 2, 1)
sns.distplot(birthweight['monpre'],
             bins = 20,
             color = 'y')

plt.xlabel('month prenatal care began')


########################

plt.subplot(2, 2, 2)
sns.distplot(birthweight['npvis'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('total number of prenatal visits')


########################

plt.subplot(2, 2, 3)

sns.distplot(birthweight['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('one minute apgar score')


########################

plt.subplot(2, 2, 4)
sns.distplot(birthweight['fmaps'],
             bins = 10,
             color = 'g')

plt.xlabel('five minutes apgar score')



plt.tight_layout()
plt.savefig('Birthweight rest.png')

plt.show()


###############################################################################

#Plotting individual relations to birthweight

###############################################################################

birthweight.columns


plt.boxplot(x='bwght',data= birthweight,meanline= True,showmeans= True)
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='monpre',data= birthweight)
plt.xlabel('Month prenatal care began')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.boxplot(x='monpre',data= birthweight,meanline= True,showmeans= True)
plt.ylabel('Month prenatal care began')
plt.show()


sns.swarmplot(x='monpre',data=birthweight)
plt.ylabel('Month prenatal care began')
plt.show()


plt.scatter(y='bwght',x='npvis',data= birthweight)
plt.xlabel('Total number of prenatal visits')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='cigs',data= birthweight)
plt.xlabel('Avg. cigarettes per day')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='drink',data= birthweight)
plt.xlabel('Avg. drinks per week')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='mage',data= birthweight)
plt.xlabel('Mother\'s age')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='fage',data= birthweight)
plt.xlabel('Father\'s age')
plt.ylabel('Birthweight (grams)')
plt.show()


plt.scatter(y='bwght',x='feduc',data= birthweight)
plt.xlabel('Mother\'s aeducation')
plt.ylabel('Birthweight (grams)')
plt.show()

###############################################################################

#Outlier flags
 
###############################################################################
    
mage_hi = 50 #after this age it is extremelly hard to conceive and there is a
             #significant drop in avg. birthweight

meduc_lo = 12 #represent high school age or lower

meduc_hi = 17 #assumed did undergrad

npvis_lo = 5 #following the 5% quantile

npvis_hi = 18 #following the 95% quantile

monpre_hi = 5 ##following the 95% quantile

fage_hi = 50 #significant drop in avg. birthweight after this

feduc_lo = 12 #high school age

feduc_hi = 17 #assumed did undergrad

cigs_lo = 0.1 #non-smokers in dataset

cigs_hi = 21 #above 95% of values

drink_lo = 4 #non-drinkers in dataset

drink_hi = 8  #Re[resent values above 80% of data


###############################################################################
# Building loops for outlier imputation


# mage

birthweight['out_mage'] = 0


for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        birthweight.loc[val[0], 'out_mage'] = 1


# meduc

birthweight['out_meduc'] = 0


for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] <= meduc_lo:
        birthweight.loc[val[0], 'out_meduc'] = -1
        

for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] >= meduc_hi:
        birthweight.loc[val[0], 'out_meduc'] = 1


# npvis

birthweight['out_npvis'] = 0


for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] <= npvis_lo:
        birthweight.loc[val[0], 'out_npvis'] = -1
        

for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] >= npvis_hi:
        birthweight.loc[val[0], 'out_npvis'] = 1
        

# monpre

birthweight['out_monpre'] = 0
        

for val in enumerate(birthweight.loc[ : , 'monpre']):
    
    if val[1] >= monpre_hi:
        birthweight.loc[val[0], 'out_monpre'] = 1


# fage

birthweight['out_fage'] = 0      

for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        birthweight.loc[val[0], 'out_fage'] = 1


# feduc

birthweight['out_feduc'] = 0


for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] < feduc_lo:
        birthweight.loc[val[0], 'out_feduc'] = -1
        

for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] > feduc_hi:
        birthweight.loc[val[0], 'out_feduc'] = 1
 
        

# cigs

birthweight['out_cigs'] = 0


for val in enumerate(birthweight.loc[ : , 'cigs']):
    
    if val[1] <= cigs_lo:
        birthweight.loc[val[0], 'out_cigs'] = -1
    
    if val[1] > cigs_hi:
        birthweight.loc[val[0], 'out_cigs'] = 1
        
#drinks

birthweight['out_drink'] = 0


for val in enumerate(birthweight.loc[ : , 'drink']):
    
    if val[1] < drink_lo:
        birthweight.loc[val[0], 'out_drink'] = -1
 
birthweight['out_drink_hi'] = 0


for val in enumerate(birthweight.loc[ : , 'drink']):       
    if val[1] > drink_hi:
       birthweight.loc[val[0], 'out_drink_hi'] = 1
       
        

#Save cleaned dataset
       
birthweight.to_excel('Birthweight_explored.xlsx')




###############################################################################

#Linear Regression Model 

###############################################################################


#Import dataset
file = 'Birthweight_explored.xlsx'

birthweight_cleaned = pd.read_excel(file)


#Choose variables and create train and test datasets 
#Note: This work only includes the variables deemed significant 

birthweight_linear = birthweight_cleaned.loc[:,['mage',
                                                'npvis',
                                                'fage',   
                                                'cigs',
                                                'drink',
                                                'out_mage',
                                                'out_meduc',
                                                'out_npvis',  
                                                'out_feduc',
                                                'out_drink_hi',
                                                'out_cigs' ]]


birthweight_target = birthweight_cleaned.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(
            birthweight_linear,
            birthweight_target,
            test_size = 0.1,
            random_state = 508)


#Combine datasets to use tool 
birthweight_train = pd.concat([X_train, y_train], axis = 1)


#Test train data in model 
lm_significant = smf.ols(formula = """bwght ~  
                                         birthweight_train['mage'] +
                                         birthweight_train['npvis'] +
                                         birthweight_train['cigs'] +
                                         birthweight_train['drink'] +
                                         birthweight_train['fage'] +
                                         birthweight_train['out_npvis'] +
                                         birthweight_train['out_cigs'] +
                                         birthweight_train['out_drink_hi'] +
                                         birthweight_train['out_feduc'] +  
                                         birthweight_train['out_mage'] +   
                                         birthweight_train['out_meduc']  
                                         """,
                                         data = birthweight_train)


# Fitting Results
results = lm_significant.fit()


# Printing Summary Statistics
print(results.summary())


#Use model on scikit learn to determine R squared 

lin_model_1 = linear_model.LinearRegression()

lin_model_1.fit(X_train, y_train)

y_pred = lin_model_1.predict(X_test)

linear_score = r2_score(y_test, y_pred).round(3)

print(linear_score)


print(f"""
Our R^2 for the model is {linear_score}.
    
This number is quite robust, but shows certain limitations with the model. The 
R^2 score is similar, but lower, than the score achieved with the training set
data.  
""")


##############################################################################

#KNN Model 

##############################################################################


"""
Since many machine learning models do not handle categorical data very well, 
we will drop categorical variables for this analysis. Fmaps and Omaps will 
also be dropped, due their lack of predictive capabilities (they are measured 
after the baby is born).
"""


birthweight_knn = birthweight_cleaned.drop(['bwght',
                                            'omaps',
                                            'fmaps',
                                            'male',
                                            'mwhte',
                                            'mblck',
                                            'moth',
                                            'fwhte',
                                            'fblck',
                                            'foth'],
                                            axis = 1)


birthweight_target = birthweight_cleaned.loc[:, 'bwght']


X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
                                                        birthweight_knn,
                                                        birthweight_target,
                                                        test_size = 0.1,
                                                        random_state = 508)

#Check values

# Training set 
print(X_train_knn.shape)
print(y_train_knn.shape)

# Testing set
print(X_test_knn.shape)
print(y_test_knn.shape)



###############################################################################
# Forming a Base for Machine Learning with KNN
###############################################################################


# Step 1: Create a model object
###############################

# How Many Neighbors?

# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []


# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train_knn, y_train_knn)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train_knn, y_train_knn))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test_knn, y_test_knn))


# Plotting the visualization
    
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# What is the optimal number of neighbors?
###########################################

print(test_accuracy)

# The best results occur when k = 4.
print(test_accuracy.index(max(test_accuracy))) # To get the actual k-value



# Building a model with k = 4
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 4)



# Fitting the model based on the training data
knn_reg.fit(X_train_knn, y_train_knn)



# Scoring the model
y_score_knn = knn_reg.score(X_test_knn, y_test_knn).round(3)

print(y_score_knn)



print(f"""
Our R^2 for the model is {y_score_knn.round(3)}.
    
This number is significantly lower than the one for the OLS mode base and 
shows a low predictive power. 
""")



###############################################################################

# Decision Trees

###############################################################################


tree_data = birthweight_cleaned.loc[:,['mage', 
                                       'npvis',
                                       'fage',   
                                       'cigs', 
                                       'drink', 
                                       'out_mage',
                                       'out_meduc',
                                       'out_npvis',  
                                       'out_feduc',
                                       'out_drink_hi',
                                       'out_cigs' ]]

tree_target = birthweight_cleaned.loc[:, 'bwght']



X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
            tree_data,
            tree_target,
            test_size = 0.1,
            random_state = 508)


tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train_tree, y_train_tree)


#Check for score 

print('Training Score', tree_full.score(X_train_tree, y_train_tree).round(4))
print('Testing Score:', tree_full.score(X_test_tree, y_test_tree).round(4))

#As the testing score was low, we will modify the model 

tree_3 = DecisionTreeRegressor(max_depth = 3,
                               random_state = 508)

tree_3_fit = tree_3.fit(X_train, y_train)


print('Training Score', tree_3.score(X_train_tree, y_train_tree).round(3))
print('Testing Score:', tree_3.score(X_test_tree, y_test_tree).round(3))

#Visual representation of tree

dot_data = StringIO()

export_graphviz(decision_tree = tree_3_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = tree_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 1200,
      width = 1600)


# Defining a function to visualize feature importance

########################
def plot_feature_importances(model, train = X_train_tree, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_tree.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
########################


# Let's plot feature importance on the full tree
        
plot_feature_importances(tree_full,
                         train = X_train_tree,
                         export = True)


#Conclusion of model 
tree_score = tree_3.score(X_test_tree, y_test_tree).round(3)


print(f"""The tree model has a score of {tree_score}.
      
This is the lowest score yet. Nevertheless the feature importance denotes how 
the predictive power of the drinks feature. This is consistent with our 
previous models.""")


###############################################################################

#Scores comparison

###############################################################################


print(f"""
      
Tree model score: {tree_score}
KNN model score: {y_score_knn}
Linear Model score: {linear_score}

The model that fits the data best is the Linear Model. This shows the robust-
ness of the method. Nevertheless, as the data shows a low linearity, there 
are other models that could be used, but they exceed the scope of the material 
studied""")


###############################################################################

#Export Predictions 

###############################################################################


y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['pred_value']

y_pred.to_excel('Birthweight_predictions.xlsx')






