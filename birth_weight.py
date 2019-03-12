# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:41:47 2019

@author: adhish
"""


# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file = 'birthweight_feature_set.xlsx'

birthweight = pd.read_excel(file)


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
        
 
# Everything will be filled with the median since the distributions are skewed 
fill = birthweight['meduc'].median()

birthweight['meduc'] = birthweight['meduc'].fillna(fill)



fill = birthweight['npvis'].median()

birthweight['npvis'] = birthweight['npvis'].fillna(fill)



fill = birthweight['feduc'].median()

birthweight['feduc'] = birthweight['feduc'].fillna(fill)


birthweight_quantiles = birthweight.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])
       
        
for col in birthweight:
    print(col)
    
########################
# Creating variables: 'page' is Total parent's age, 'pduc' is total parent's education
# and 'mon' is the number of months that the mothers was under prenatal care 
########################

birthweight['page'] = birthweight.mage + birthweight.fage

birthweight['pduc'] = birthweight.meduc + birthweight.feduc

birthweight['mon'] =  9 - birthweight.monpre

birthweight.head()

########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(birthweight['mage'],
             bins = 35,
             color = 'g')

plt.xlabel('mage')


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['meduc'],
             bins = 30,
             color = 'y')

plt.xlabel('meduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthweight['monpre'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('monpre')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthweight['npvis'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('npvis')



plt.tight_layout()
plt.savefig('birthweight Data Histograms 1 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthweight['fage'],
             bins = 35,
             color = 'g')

plt.xlabel('fage')


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['feduc'],
             bins = 30,
             color = 'y')

plt.xlabel('feduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthweight['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('omaps')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthweight['fmaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fmaps')



plt.tight_layout()
plt.savefig('birthweight Data Histograms 2 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthweight['cigs'],
             bins = 30,
             color = 'y')

plt.xlabel('cigs')



########################

plt.subplot(2, 2, 2)
sns.distplot(birthweight['drink'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('drink')



########################

plt.subplot(2, 2, 3)

sns.distplot(birthweight['male'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('male')



########################

plt.subplot(2, 2, 4)
sns.distplot(birthweight['mwhte'],
             bins = 35,
             color = 'g')

plt.xlabel('mwhte')



plt.tight_layout()
plt.savefig('birthweight Data Histograms 3 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthweight['mblck'],
             bins = 30,
             color = 'y')

plt.xlabel('mblck')



########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['moth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('moth')



########################

plt.subplot(2, 2, 3)

sns.distplot(birthweight['fwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fwhte')



plt.subplot(2, 2, 4)
sns.distplot(birthweight['fblck'],
             bins = 35,
             color = 'g')

plt.xlabel('fblck')



plt.tight_layout()
plt.savefig('birthweight Data Histograms 4 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)

sns.distplot(birthweight['foth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('foth')



plt.subplot(2, 2, 2)

sns.distplot(birthweight['bwght'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('bwght')

plt.subplot(2, 2, 2)

sns.distplot(birthweight['bad'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('bad')

plt.subplot(2, 2, 2)

sns.distplot(birthweight['meduc'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('meduc')


plt.tight_layout()
plt.savefig('birthweight Data Histograms 5 of 5.png')

plt.show()

#Looking at distribution of the new variables that we created above 

sns.distplot(birthweight['mon'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('mon')
plt.tight_layout()
plt.show


sns.distplot(birthweight['page'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('page')
plt.tight_layout()
plt.show


sns.distplot(birthweight['pduc'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('pduc')
plt.tight_layout()
plt.show


########################
# Creating Outlier Flags
########################
mage_low = 20
mage_high = 55
meduc_low = 10
meduc_high = 16
monpre_high = 5
npvis_low = 5
npvis_high = 17
fage_low = 20
fage_high = 50
feduc_low = 10
feduc_high = 16
omaps_low = 7
omaps_high = 10
fmaps_low = 8
fmaps_high = 10
cigs_high = 17
drink_low = 4
drink_high = 10
bwght_low =2000
bwght_high = 4500
bad_low = 12
bad_high = 20
page_high= 95
pduc_low=18
pduc_high=28
mon_high=6


# Building loops for outlier imputation

########################
# Mother Age

birthweight['out_mage'] = 0


for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] <= mage_low:
        birthweight.loc[val[0], 'out_mage'] = -1



for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] >= mage_high:
        birthweight.loc[val[0], 'out_mage'] = 1

########################
# Mother's eductcation
#print(birthweight['out_meduc'].isnull().sum())
birthweight['out_meduc'] = 0


for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] <= meduc_low:
        birthweight.loc[val[0], 'out_meduc'] = -1



for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] >= meduc_high:
        birthweight.loc[val[0], 'out_meduc'] = 1
        
########################
# Monthly prenatal visits


birthweight['out_monpre'] = 0 

for val in enumerate(birthweight.loc[ : , 'monpre']):
    
    if val[1] >= monpre_high:
        birthweight.loc[val[0], 'out_monpre'] = 1
       
########################
# npvis

birthweight['out_npvis'] = 0


for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] <= npvis_low:
        birthweight.loc[val[0], 'out_npvis'] = -1



for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] >= npvis_high:
        birthweight.loc[val[0], 'out_npvis'] = 1
        
        
            
########################
# father's age

birthweight['out_fage'] = 0


for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] <= fage_low:
        birthweight.loc[val[0], 'out_fage'] = -1



for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] >= fage_high:
        birthweight.loc[val[0], 'out_fage'] = 1

########################
# father's education

birthweight['out_feduc'] = 0


for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] <= feduc_low:
        birthweight.loc[val[0], 'out_feduc'] = -1



for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] >= feduc_high:
        birthweight.loc[val[0], 'out_feduc'] = 1

########################
# omaps 

birthweight['out_omaps'] = 0


for val in enumerate(birthweight.loc[ : , 'omaps']):
    
    if val[1] <= omaps_low:
        birthweight.loc[val[0], 'out_omaps'] = -1



for val in enumerate(birthweight.loc[ : , 'omaps']):
    
    if val[1] >= omaps_high:
        birthweight.loc[val[0], 'out_omaps'] = 1

########################
# fmaps 

birthweight['out_fmaps'] = 0


for val in enumerate(birthweight.loc[ : , 'fmaps']):
    
    if val[1] <= fmaps_low:
        birthweight.loc[val[0], 'out_fmaps'] = -1



for val in enumerate(birthweight.loc[ : , 'fmaps']):
    
    if val[1] >= fmaps_high:
        birthweight.loc[val[0], 'out_fmaps'] = 1
        
########################
# cigs_high


birthweight['out_cigs'] = 0 

for val in enumerate(birthweight.loc[ : , 'cigs']):
    
    if val[1] >= cigs_high:
        birthweight.loc[val[0], 'out_cigs'] = 1
########################
# drink 

birthweight['out_drink'] = 0


for val in enumerate(birthweight.loc[ : , 'drink']):
    
    if val[1] <= drink_low:
        birthweight.loc[val[0], 'out_fmaps'] = -1



for val in enumerate(birthweight.loc[ : , 'drink']):
    
    if val[1] >= drink_high:
        birthweight.loc[val[0], 'out_drink'] = 1

########################
# bwght 

birthweight['out_bwght'] = 0


for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] <= bwght_low:
        birthweight.loc[val[0], 'out_bwght'] = -1



for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] >= bwght_high:
        birthweight.loc[val[0], 'out_bwght'] = 1

########################
# page
        
page_high= 95
birthweight['out_page'] = 0

for val in enumerate(birthweight.loc[ : , 'page']):
    
    if val[1] >= page_high:
        birthweight.loc[val[0], 'out_page'] = 1
        

########################
# pduc
        
birthweight['out_pduc'] = 0

for val in enumerate(birthweight.loc[ : , 'pduc']):
    
    if val[1] >= pduc_high:
        birthweight.loc[val[0], 'out_pduc'] = 1
        

for val in enumerate(birthweight.loc[ : , 'pduc']):
    
    if val[1] <= pduc_low:
        birthweight.loc[val[0], 'out_pduc'] = -1


########################
# mon

birthweight['out_mon']=0

for val in enumerate(birthweight.loc[ : , 'mon']):
    
    if val[1] >= mon_high:
        birthweight.loc[val[0], 'out_mon'] = 1


###############################################################################
# Qualitative Variable Analysis (Boxplots)
###############################################################################
        
"""

Assumed Categorical -

male
mwhte
mblck
moth
fwhte
fblck
foth
"""

########################
# mwhte


birthweight.boxplot(column = ['bwght'],
                by = ['mwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by mother white")
plt.suptitle("")

plt.savefig("baby weight by mother white.png")

plt.show()

########################
# mblck


birthweight.boxplot(column = ['bwght'],
                by = ['mblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by mother black")
plt.suptitle("")

plt.savefig("baby weight by mother black.png")

plt.show()

########################
# moth


birthweight.boxplot(column = ['bwght'],
                by = ['moth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by mother other")
plt.suptitle("")

plt.savefig("baby weight by mother other.png")

plt.show()

########################
# fwhte


birthweight.boxplot(column = ['bwght'],
                by = ['fwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by father white")
plt.suptitle("")

plt.savefig("baby weight by father white.png")

plt.show()

########################
# fblck


birthweight.boxplot(column = ['bwght'],
                by = ['fblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by father black")
plt.suptitle("")

plt.savefig("baby weight by father black.png")

plt.show()

########################
# foth


birthweight.boxplot(column = ['bwght'],
                by = ['foth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by father other")
plt.suptitle("")

plt.savefig("baby weight by father other.png")

plt.show()

########################
# mwhte+mblck+moth

birthweight.boxplot(column = ['bwght'],
                by = ['mwhte']+['mblck']+['moth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("baby weight by mother white")
plt.suptitle("")

plt.savefig("baby weight by mother white.png")

plt.show()

### 3rd Boxplot - black parents

birthweight['blackparents'] = birthweight.mblck + birthweight.fblck
    
birthweight.blackparents [birthweight.blackparents < 2 ] = 0
birthweight.blackparents [birthweight.blackparents >= 2] = 1
    

birthweight.boxplot(column = ['bwght'],
                by = ['blackparents'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("bwght by blackparents")
plt.suptitle("")

plt.savefig("bwght by blackparents.png")

plt.show()

# We see no difference with baby that has both black parents. This does not 
# reflect one of our findings online that black parents' babies are normally have 
# less weight than other babies.

###############################################################################
# Correlation Analysis
###############################################################################

birthweight.head()


df_corr = birthweight.corr().round(2)


print(df_corr)


df_corr.loc['bwght'].sort_values(ascending = False)


########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:, 1:]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('birthweight Correlation Heatmap.png')
plt.show()

birthweight.to_excel('birthweight_explored_final.xlsx')


###############################################################################
# Statsmodel 
###############################################################################

# Loading Libraries
import pandas as pd
import statsmodels.formula.api as smf # regression modeling
import seaborn as sns
import matplotlib.pyplot as plt


file = 'birthweight_explored_final.xlsx'

birthweight_3 = pd.read_excel(file)

birthweight_3.columns

# Building a Regression 
lm_price_qual = smf.ols(formula = """bwght ~ birthweight_3['mage']+
                                        birthweight_3['meduc']+
                                        birthweight_3['monpre']+
                                        birthweight_3['mon']+
                                        birthweight_3['out_mage']+
                                        birthweight_3['npvis']+
                                        birthweight_3['fage']+
                                        birthweight_3['page']+
                                        birthweight_3['feduc']+
                                        birthweight_3['pduc']+
                                        birthweight_3['omaps']+
                                        birthweight_3['fmaps']+
                                        birthweight_3['cigs']+
                                        birthweight_3['drink']+
                                        birthweight_3['male']+
                                        birthweight_3['mwhte']+
                                        birthweight_3['mblck']+
                                        birthweight_3['moth']+
                                        birthweight_3['fwhte']+
                                        birthweight_3['fblck']+
                                        birthweight_3['foth']+
                                        birthweight_3['out_npvis']+
                                        birthweight_3['out_fage']+
                                        birthweight_3['out_feduc']+
                                        birthweight_3['out_omaps']+
                                        birthweight_3['out_fmaps']+
                                        birthweight_3['out_cigs']+
                                        birthweight_3['out_monpre']+
                                        birthweight_3['out_drink']+
                                        birthweight_3['out_pduc']+
                                        birthweight_3['out_mon']+
                                        birthweight_3['out_page'] -1
                                        """,
                                        data = birthweight_3)



# Fitting Results
results = lm_price_qual.fit()


# Printing Summary Statistics
print(results.summary())


print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    
###############################################################################
# KNN model 
###############################################################################
# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling

birthweight_data = birthweight_3.loc[:,['monpre',
                                       'mon',
                                       'mage',
                                       'pduc',
                                       'cigs',
                                       'drink',
                                       'out_page',
                                       'out_drink',
                                       ]]


birthweight_target = birthweight_3.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.3,
            random_state = 508)


# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []



# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    

# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#print(testaccuracy.index(max(test_accuracy))) -> to index lists
print(test_accuracy.index(max(test_accuracy))) 


# The best results occur when k = 8.


# Building a model with k = 8
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 8)



# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)



# Scoring the model
y_score = knn_reg.score(X_test, y_test)
y1_score= knn_reg.score(X_train,y_train)


# The score is directly comparable to R-Square
print(y1_score) #train
print(y_score) #test 


###############################################################################
# OLS Regression Analysis 
###############################################################################

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.3,
            random_state = 508)



# Prepping the Model
lr = LinearRegression(fit_intercept = False)


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)

#Saving predictions to excel since OLS has the highest y_score
prediction=pd.DataFrame(lr_pred)
prediction.to_excel('Prediction.xlsx')

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score of OLS 
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))


#Comparing 2 model results
print(f"""
Optimal model KNN score: {y_score.round(2)}
Optimal model OLS score: {y_score_ols_optimal.round(2)}
""")


###############################################################################
# Decision Trees
###############################################################################


# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split

# Building a full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))



# Creating a tree with only two levels.
tree_2 = DecisionTreeRegressor(max_depth = 2,
                               random_state = 508)

tree_2_fit = tree_2.fit(X_train, y_train)


print('Training Score', tree_2.score(X_train, y_train).round(4))
print('Testing Score:', tree_2.score(X_test, y_test).round(4))


# Our tree is much less predictive, but it can be graphed in a way that is
# easier to interpret.

dot_data = StringIO()


export_graphviz(decision_tree = tree_2_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = birthweight_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)


tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))



# Defining a function to visualize feature importance

########################
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
########################


plot_feature_importances(tree_leaf_50,
                         train = X_train,
                         export = True)



# Let's plot feature importance on the full tree.
plot_feature_importances(tree_full,
                         train = X_train,
                         export = False)

#Conclusion 
# As we can see from the 3 models, our best prediction model was our OLS model
# with the y_score of 0.7

