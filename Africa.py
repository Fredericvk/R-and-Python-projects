# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

###############################################################################
# Importing libraries and base dataset
###############################################################################

import pandas as pd # data science essentials (read_excel, DataFrame)
import matplotlib.pyplot as plt # data visualization

file = 'world_data_hult_regions.xlsx'
data = pd.read_excel(file)

data.describe()
data.info()

#slicing the dataset so that we only got the countries in our region
#to a variable called africa

africa = data[data['Hult_Team_Regions'] == 'Central Africa 2']

africa.head()
africa.describe()
africa.shape

###############################################################################
# Missing Values
###############################################################################

africa.isnull().sum()

#create a loop to flag all missing values

for col in africa:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if africa[col].isnull().any():
        africa['a_'+col] = africa[col].isnull().astype(int)

#replace all missing values with the median of the column
        
africa_median  = pd.DataFrame.copy(africa)

#create a loop to replace all missing values with the median
for col in africa_median:
    
    """ Impute missing values using the mean of each column """
    
    if africa_median[col].isnull().any():
        
        col_median = africa_median[col].median()
        
        africa_median[col] = africa_median[col].fillna(col_median).round(2)
        
africa_median.isnull().sum()

###############################################################################
#OUtliers
###############################################################################

#employment outliers
africa_median.boxplot(column = ['pct_female_employment', 
                                'pct_male_employment', 
                                'pct_agriculture_employment',
                                'pct_industry_employment',
                                'pct_services_employment',
                                ],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )


plt.title("Boxplots for compulsory_edu_yrs, pct_female_employment, pct_male_employment, pct_agriculture_employment, pct_industry_employment, pct_services_employment and unemployment_pct")

plt.show()

#industry employment outliers
africa_median.boxplot(column = ['pct_industry_employment'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_industry_employment = africa_median[africa_median['pct_industry_employment']>16]
print(outliers_industry_employment['country_name'])

#unemployment outliers
africa_median.boxplot(column = ['unemployment_pct'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

plt.title("unemployment_pct")

plt.show()

#low unemployment
outliers_unemployment = africa_median[africa_median['unemployment_pct'] < 1.5]
print(outliers_unemployment['country_name'])

print(africa_median.loc[:,["country_name","unemployment_pct"]])

#high unemployment
outliers_unemployment_high = africa_median[africa_median['unemployment_pct'] > 8]
print(outliers_unemployment_high['country_name'])


#compulsory education years
africa_median.boxplot(column = ['compulsory_edu_yrs'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_edu = africa_median[africa_median['compulsory_edu_yrs'] == 6]
print(outliers_edu['country_name'])

plt.title("compulsory_edu_years")

plt.show()

print(africa_median.loc[:,["country_name","compulsory_edu_yrs"]])

#gdp outliers
africa_median.boxplot(column = ['exports_pct_gdp',
                                'fdi_pct_gdp',
                                'gdp_growth_pct'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_fdi_pct_gdp = africa_median[africa_median['fdi_pct_gdp'] > 8]
print(outliers_fdi_pct_gdp['country_name'])

#energy
africa_median.boxplot(column = ['access_to_electricity_pop',
                                'access_to_electricity_rural',
                                'access_to_electricity_urban',
                                ],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_elr = africa_median[africa_median['access_to_electricity_rural'] > 30]
print(outliers_elr['country_name'])

#CO2 Emission
africa_median.boxplot(column = ['CO2_emissions_per_capita)'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_co2 = africa_median[africa_median['CO2_emissions_per_capita)'] > 0.6]
print(outliers_co2['country_name'])

#air pollution
africa_median.boxplot(column = ['avg_air_pollution'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_ap = africa_median[africa_median['avg_air_pollution'] > 80]
print(outliers_ap['country_name'])

#population
africa_median.boxplot(column = ['urban_population_growth_pct'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_growth = africa_median[africa_median['urban_population_growth_pct'] > 4.5]
print(outliers_growth['country_name'])

outliers_growthL = africa_median[africa_median['urban_population_growth_pct'] == 1]
print(outliers_growthL['country_name'])

africa_median.boxplot(column = ['urban_population_pct'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

#Social
africa_median.boxplot(column = ['internet_usage_pct'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )

outliers_internet = africa_median[africa_median['internet_usage_pct'] > 14]
print(outliers_internet['country_name'])

###############################################################################
#Outlier analysis
###############################################################################
import seaborn as sns

#Mauritania
#CO2

plt.subplot(3, 1, 1)
sns.distplot(africa_median['CO2_emissions_per_capita)'],
             bins = 'fd',
             color = 'r')

plt.title('Mauritania outliers')
plt.xlabel('CO2 emissions per capita')
plt.tight_layout()
plt.axvline(x = 0.66,
            label = 'Mauritania',
            linestyle = '--')

#Air Pollution
plt.subplot(3, 1, 2)
sns.distplot(africa_median['avg_air_pollution'],
             bins = 'fd',
             color = 'orange')

plt.xlabel('Average air pollution')
plt.tight_layout()
plt.axvline(x = 82.44,
            label = 'Mauritania',
            linestyle = '--')

#FDI
plt.subplot(3, 1, 3)
sns.distplot(africa_median['fdi_pct_gdp'],
             bins = 'fd',
             color = 'y')

plt.xlabel('% FDI of GDP')
plt.tight_layout()
plt.axvline(x = 9.32,
            label = 'Mauritania',
            linestyle = '--')

#Benin
#% Inudstry employment
plt.subplot(2, 1, 1)
sns.distplot(africa_median['pct_industry_employment'],
             bins = 'fd',
             color = 'r')

plt.title('Benin outliers')
plt.xlabel('% Industry employment')
plt.tight_layout()
plt.axvline(x = 18.76,
            label = 'Benin',
            linestyle = '--')

#% CO2 emissions per capita
plt.subplot(2, 1, 2)
sns.distplot(africa_median['CO2_emissions_per_capita)'],
             bins = 'fd',
             color = 'orange')

plt.xlabel('CO2 emissions per capita')
plt.tight_layout()
plt.axvline(x = 0.6142,
            label = 'Mauritania',
            linestyle = '--')


###############################################################################
#Exploratory Analysis
###############################################################################
import seaborn as sns # more data visualization

#correlation GDP - Employment

subset = ['unemployment_pct',
          'pct_female_employment', 
          'pct_male_employment', 
          'pct_agriculture_employment',
          'pct_industry_employment',
          'pct_services_employment',
          'compulsory_edu_yrs',
          'gdp_growth_pct',
          'fdi_pct_gdp',
          'exports_pct_gdp',
          'gdp_usd']

subset3 = ['incidence_hiv',
           'child_mortality_per_1k',
           'women_in_parliament']

africa_corr = africa_median[subset3].corr().round(2)

sns.heatmap(africa_corr,
            cmap = 'Reds',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5,
            cbar = True)

#Correlation Electricity - Air Pollution - GDP

subset2 = ['access_to_electricity_pop', 
          'access_to_electricity_rural', 
          'access_to_electricity_urban',
          'internet_usage_pct',
          'pct_agriculture_employment',
          'pct_industry_employment',
          'pct_services_employment',
          'unemployment_pct',
          'fdi_pct_gdp',
          'exports_pct_gdp',
          'gdp_growth_pct',
          'gdp_usd']

africa_corr2 = africa_median[subset2].corr().round(2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(africa_corr2,
            cmap = 'Reds',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5,
            cbar = False)

plt.savefig("heatmap")

#Relation internet usage pct - access to electricity pop
sns.lmplot(x = 'access_to_electricity_pop',
           y = 'internet_usage_pct',
           data = africa_median)
plt.title("Relation internet usage & access to electricity")
plt.grid()
plt.tight_layout()
plt.xlabel("% population access to electricity")
plt.ylabel("% internet usage")
plt.savefig("access to electricity - internet usage")

plt.show()


sns.lmplot(x = 'internet_usage_pct',
           y = 'pct_services_employment',
           fit_reg = True,
           data = africa_median)
plt.title("Relation services employment & internet usage")
plt.grid()
plt.tight_layout()
plt.xlabel("% internet usage")
plt.ylabel("% services employment")
plt.savefig("internet usage - services employment")
plt.show()


sns.lmplot(x = 'pct_services_employment',
           y = 'exports_pct_gdp',
           fit_reg = True,
           data = africa_median)
plt.title("Relation % exports of GDP & services employment")
plt.grid()
plt.tight_layout()
plt.xlabel("% services employment")
plt.ylabel("% exports of GDP")
plt.savefig("exports of GDP - services employment")
plt.show()

#trials

sns.lmplot(x = 'access_to_electricity_pop',
           y = 'exports_pct_gdp',
           data = africa_median,
           fit_reg = False,
           hue = 'country_name',
           scatter_kws= {"marker": "D", 
                        "s": 30},
           truncate = False,
           palette = 'plasma')

plt.title("Relation % exports of GDP & access to electricity")
plt.tight_layout()
plt.xlabel("% population access to electricity")
plt.ylabel("% exports of GDP")
plt.savefig("relation exports & electricity")

print(africa_median[africa_median['exports_pct_gdp']>=40]['country_name'])

#Chad and South Sudan export a lot of oil
#exclude Chad and south sudan

africa_median2 = africa_median[africa_median['access_to_electricity_pop']>=40]

sns.lmplot(x = 'access_to_electricity_pop',
           y = 'exports_pct_gdp',
           data = africa_median2,
           fit_reg = True,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           truncate = False,
           palette = 'plasma')

#relation electricity - gdp growth

sns.lmplot(x = 'access_to_electricity_urban',
           y = 'gdp_growth_pct',
           data = africa_median,
           hue = 'country_name',
           fit_reg = False,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           palette = 'plasma')



plt.title("Relation pct_industry_employment & pct_services_employemnt per country")
plt.grid()
plt.tight_layout()
plt.show()


sns.lmplot(x = 'exports_pct_gdp',
           y = 'pct_industry_employment',
           data = africa_median,
           hue = 'country_name',
           fit_reg = False,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           palette = 'plasma')


plt.title("Relation pct_industry_employment & pct_services_employemnt per country")
plt.grid()
plt.tight_layout()
plt.show()

sns.lmplot(x = 'pct_agriculture_employment',
           y = 'pct_services_employment',
           data = africa_median,
           hue = 'country_name',
           fit_reg = False,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           palette = 'plasma')


plt.title("Relation pct_industry_employment & pct_services_employemnt per country")
plt.grid()
plt.tight_layout()
plt.show()

sns.lmplot(x = 'pct_male_employment',
           y = 'gdp_growth_pct',
           data = africa_median,
           hue = 'country_name',
           fit_reg = False,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           palette = 'plasma')


plt.title("Relation pct_industry_employment & pct_services_employemnt per country")
plt.grid()
plt.tight_layout()
plt.show()

sns.lmplot(x = 'pct_services_employment',
           y = 'exports_pct_gdp',
           data = africa_median,
           fit_reg = True,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           palette = 'plasma')


plt.title("Relation services employment & exports % of GDP")
plt.xlabel(")
plt.grid()
plt.tight_layout()
plt.show()


sns.violinplot(x = 'access_to_electricity_pop',
               y = 'exports_pct_gdp',
               data = africa_median)

plt.show()



sns.stripplot(x = 'access_to_electricity_pop',
              y = 'exports_pct_gdp',
              data = africa_median,
              size = 5,
              orient = 'v')
plt.show()



sns.swarmplot(x = 'access_to_electricity_pop',
              y = 'exports_pct_gdp',
              data = africa_median,
              size = 5,
              orient = 'v')

plt.show()

###############################################################################
#comparative Analysis
###############################################################################
df.loc[df['B'].isin(['one','three'])]

africaw = data.loc[data['Hult_Team_Regions'].isin(["Central Africa 2", "World"])]

for col in africa12:
    
    """ Impute missing values using the mean of each column """
    
    if africa12[col].isnull().any():
        
        col_median = africa12[col].median()
        
        africa12[col] = africa12[col].fillna(col_median).round(2)

sns.lmplot(x = 'access_to_electricity_pop',
           y = 'exports_pct_gdp',
           data = africaw,
           fit_reg = False,
           hue = 'country_name',
           scatter_kws= {"marker": "D", 
                        "s": 30},
           truncate = False,
           palette = 'plasma')
plt.title("% exports of GDP vs access to electricity")
plt.savefig("world ate")

plt.subplot(2, 1, 1)
africaw.boxplot(column = ['access_to_electricity_pop'],
                 vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,
                 )
plt.title("boxplot: access to electricity")

plt.subplot(2, 1, 2)
sns.distplot(africaw['access_to_electricity_pop'],
             bins = 'fd',
             color = 'r')

plt.title('Distribution access to electricity')
plt.xlabel('% population that has access to electricity')
plt.tight_layout()
plt.axvline(x = 85.7,
            label = 'World',
            linestyle = '--')

plt.savefig("ate")

africa12 = data.loc[data['Hult_Team_Regions'].isin(["Central Africa 2", "Central Aftica 1"])]
        

subset4 = ['access_to_electricity_pop',
          'internet_usage_pct',
          'pct_services_employment',
          'exports_pct_gdp',
          'gdp_growth_pct']

africa_corr4 = africa12[subset4].corr().round(2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(africa_corr4,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5,
            cbar = False)
plt.title("heatmap Central Africa 1 & 2")

plt.savefig("heatmap africa 1 & 2")

sns.lmplot(x = 'access_to_electricity_pop',
           y = 'exports_pct_gdp',
           data = africa12,
           fit_reg = True,
           scatter_kws= {"marker": "D", 
                        "s": 30},
           truncate = False,
           palette = 'plasma')

plt.title("Relation % exports of GDP & access to electricity")
plt.tight_layout()
plt.xlabel("% population access to electricity")
plt.ylabel("% exports of GDP")
plt.savefig("axe")