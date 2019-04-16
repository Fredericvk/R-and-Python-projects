#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:30:30 2019

@author: fredericvankelecom
"""

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans # k-means clustering

# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

survey_df = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')

###############################################################################

# MODEL CODE

###############################################################################


###############################################################################
# PCA
###############################################################################

########################
# Step 1: Remove demographic information
########################

#we drop age (q1), device (q2), education (q48), marital status (q49),
#children (q50r1 - q50r5), race (q54), latino (q55), income (q56) and gender (q57)

survey_features_reduced = survey_df.iloc[:, 12:77]

########################
# Step 2: Scale the behavioural variables to get equal variance
########################

scaler = StandardScaler()


scaler.fit(survey_features_reduced)


X_scaled_reduced = scaler.transform(survey_features_reduced)

########################
# Step 3: Run PCA without limiting the number of components
########################

survey_pca_reduced = PCA(n_components = None,
                         random_state = 508)


survey_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)


########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(survey_pca_reduced.n_components_)


plt.plot(features,
         survey_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()

#Based on the graph, I decided to go with 4 PCA's.

########################
# Step 5: Run PCA again based on the desired number of components
########################

survey_pca_reduced = PCA(n_components = 4,
                           random_state = 508)


survey_pca_reduced.fit(X_scaled_reduced)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(survey_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(survey_df.columns[12:77])


print(factor_loadings_df)


factor_loadings_df.to_excel('factor_loadings.xlsx')



########################
# Step 7: Analyze factor strengths per survey taker
########################

X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)



########################
# Step 8: Rename your principal components and reattach demographic information
########################

X_pca_df.columns = ['PCA 1',
                    'FOCUS',
                    'PCA 2',
                    'PCA 4']

###############################################################################
# Combining PCA and Clustering
###############################################################################

########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))

########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns

########################
# Step 3: Experiment with different numbers of clusters
########################


survey_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


survey_k_pca.fit(X_pca_clust_df)


survey_kmeans_pca = pd.DataFrame({'cluster': survey_k_pca.labels_})


print(survey_kmeans_pca.iloc[: , 0].value_counts())

########################
# Plot Inertia
########################

ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)


    # Fit model to samples
    model.fit(X_pca_clust_df)


    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



# Plot ks vs inertias
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)


plt.show()

# Based on the inertia plot and the value counts, I decided to go with 5 clusters

########################
# Step 4: Analyze cluster centers
########################

centroids_pca = survey_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['PCA1', 
                            'FOCUS', 
                            'PCA3',
                            'PCA4']



print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################


final_pca_clust_df = pd.concat([survey_df.loc[ : , ['q1',
                                                   'q2r1',
                                                   'q2r2',
                                                   'q2r3',
                                                   'q2r4',
                                                   'q2r5',
                                                   'q2r6',
                                                   'q2r7',
                                                   'q2r8',
                                                   'q2r9',
                                                   'q2r10',
                                                   'q48',
                                                   'q49',
                                                   'q50r1',
                                                   'q50r2',
                                                   'q50r3',
                                                   'q50r4',
                                                   'q50r5',
                                                   'q54',
                                                   'q55',
                                                   'q56',
                                                   'q57',
                                                   ]] , 
                                                    clst_pca_df], 
                                                    axis = 1)

final_pca_clust_df.rename(columns = {'q1':'Age',
                                    'q2r1':'iPhone',
                                    'q2r2':'iPod Touch',
                                    'q2r3':'Android',
                                    'q2r4':'Blackberry',
                                    'q2r5':'Nokia',
                                    'q2r6':'Windows phone',
                                    'q2r7':'HP/Palm',
                                    'q2r8':'Tablet',
                                    'q2r9':'Other smartphone',
                                    'q2r10':'None',
                                    'q48':'Education',
                                    'q49':'Marital status',
                                    'q50r1':'No children',
                                    'q50r2':'Children under 6',
                                    'q50r3':'Children between 6-12',
                                    'q50r4':'Children between 13-17',
                                    'q50r5':'Children over 18',
                                    'q54':'Race',
                                    'q55':'Latino_Hispanic',
                                    'q56':'Income',
                                    'q57':'Gender'},
                                     inplace = True)

# Renaming ages
ages = {1 : 'under 18',
        2 : '18-24',
        3 : '25-29',
        4 : '30-34',
        5 : '35-39',
        6 : '40-44',
        7 : '45-49',
        8 : '50-54',
        9 : '55-59',
        10 : '60-64',
        11 : '65+'}

final_pca_clust_df['Age'] = final_pca_clust_df['Age'].replace(ages, inplace = False)

# Renaming devices
yes_no = {0 : 'no',
          1 : 'yes'}

final_pca_clust_df['iPhone'] = final_pca_clust_df['iPhone'].replace(yes_no, inplace = False)
final_pca_clust_df['iPod Touch'] = final_pca_clust_df['iPod Touch'].replace(yes_no, inplace = False)
final_pca_clust_df['Android'] = final_pca_clust_df['Android'].replace(yes_no, inplace = False)
final_pca_clust_df['Blackberry'] = final_pca_clust_df['Blackberry'].replace(yes_no, inplace = False)
final_pca_clust_df['Nokia'] = final_pca_clust_df['Nokia'].replace(yes_no, inplace = False)
final_pca_clust_df['Windows phone'] = final_pca_clust_df['Windows phone'].replace(yes_no, inplace = False)
final_pca_clust_df['HP/Palm'] = final_pca_clust_df['HP/Palm'].replace(yes_no, inplace = False)
final_pca_clust_df['Tablet'] = final_pca_clust_df['Tablet'].replace(yes_no, inplace = False)
final_pca_clust_df['Other smartphone'] = final_pca_clust_df['Other smartphone'].replace(yes_no, inplace = False)
final_pca_clust_df['None'] = final_pca_clust_df['None'].replace(yes_no, inplace = False)

# Renaming education
education = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some college',
             4 : 'College graduate',
             5 : 'Some post-graduate studies ',
             6 : 'Post graduate degree'}


final_pca_clust_df['Education'] = final_pca_clust_df['Education'].replace(education, inplace = False)

# Renaming marital status
marital_status = {1 : 'Married',
                  2 : 'Single',
                  3 : 'Single with a partner',
                  4 : 'Separated/Widowed/Divorced'}


final_pca_clust_df['Marital status'] = final_pca_clust_df['Marital status'].replace(marital_status, inplace = False)

# Renaming no children
final_pca_clust_df['No children'] = final_pca_clust_df['No children'].replace(yes_no, inplace = False)
final_pca_clust_df['Children under 6'] = final_pca_clust_df['Children under 6'].replace(yes_no, inplace = False)
final_pca_clust_df['Children between 6-12'] = final_pca_clust_df['Children between 6-12'].replace(yes_no, inplace = False)
final_pca_clust_df['Children between 13-17'] = final_pca_clust_df['Children between 13-17'].replace(yes_no, inplace = False)
final_pca_clust_df['Children over 18'] = final_pca_clust_df['Children over 18'].replace(yes_no, inplace = False)

# Renaming race
race = {1 : 'White or Caucasian',
        2 : 'Black or African American',
        3 : 'Asian',
        4 : 'Native Hawaiian or Other Pacific Islander',
        5 : 'American Indian or Alaska Native',
        6 : 'Other race'}


final_pca_clust_df['Race'] = final_pca_clust_df['Race'].replace(race, inplace = False)

# Renaming hispanic or latino
yes_no2 = {1 : 'yes',
           2 : 'no'}

final_pca_clust_df['Latino_Hispanic'] = final_pca_clust_df['Latino_Hispanic'].replace(yes_no2, inplace = False)

# Renaming income
income = {1 : 'Under $10,000',
          2 : '$10,000-$14,999',
          3 : '$15,000-$19,999',
          4 : '$20,000-$29,999',
          5 : '$30,000-$39,999',
          6 : '$40,000-$49,999',
          7 : '$50,000-$59,999',
          8 : '$60,000-$69,999',
          9 : '$70,000-$79,999',
          10 : '$80,000-$89,999',
          11 : '$90,000-$99,999',
          12 : '$100,000-$124,999',
          13 : '$125,000-$149,999',
          14 : '$150,000 and over'}


final_pca_clust_df['Income'] = final_pca_clust_df['Income'].replace(income, inplace = False)

# Renaming gender
gender = {1 : 'male',
          2 : 'female'}


final_pca_clust_df['Gender'] = final_pca_clust_df['Gender'].replace(gender, inplace = False)

final_pca_clust_df.to_excel('exam.xlsx')

###############################################################################

# CODE FOR DATA ANALYSIS

###############################################################################

#What people are in cluster 0?

cluster_0 = final_pca_clust_df[final_pca_clust_df['cluster'] == 0]

# Analyzing by Age
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['Age'],
            y = cluster_0['FOCUS'])

cluster_0['Age'].value_counts()

# Analyzing by Education
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['Education'],
            y = cluster_0['FOCUS'])

cluster_0['Education'].value_counts()

# Analyzing by Marital status
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['Marital status'],
            y = cluster_0['FOCUS'])

cluster_0['Marital status'].value_counts()

# Analyzing by Children
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['No children'],
            y = cluster_0['FOCUS'])

cluster_0['No children'].value_counts()

# Analyzing by Race
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['Race'],
            y = cluster_0['FOCUS'])

cluster_0['Race'].value_counts()

# Analyzing by Income
fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(x = cluster_0['Race'],
            y = cluster_0['FOCUS'])

cluster_0['Income'].value_counts()

#Devices
#iPhone
cluster_0['iPhone'].value_counts()
df = cluster_0[cluster_0['iPhone'] == 'no'] 
df1 = df[df['iPod Touch'] == 'yes']

cluster_0['Gender'].value_counts()


#Android
cluster_0[cluster_0['Android'] == 1 & cluster_0['iPhone'] == 0]['Android'].value_counts()
df = cluster_0[cluster_0['Android'] == 'yes'] 
df1 = df[(df['iPhone'] == 'no')]
df2 = df1[df1['iPod Touch'] == 'no']
df1['Android'].value_counts()
#Blackberry
cluster_0['Blackberry'].value_counts()

