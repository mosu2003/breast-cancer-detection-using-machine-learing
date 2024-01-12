
# Breast Cancer Detection

### Import ML packages
"""
# data manipulation (NumPy, Pandas),
#  data visualization (Matplotlib, Seaborn)
# machine learning algorithms (scikit-learn)
# The specific scikit-learn modules include classification metrics, model selection tools
# various classifiers such as Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine (SVM).


# Commented out IPython magic to ensure Python compatibility.
import numpy as np # Imports the NumPy library and aliases it as np
import pandas as pd # Imports the Pandas library and aliases it as pd
import matplotlib.pyplot as plt # Imports the pyplot module from Matplotlib and aliases it as plt
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score#Imports specific functions for classification metrics from scikit-learn.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold #Imports functions for data splitting, cross-validation, and k-fold cross-validation.
from sklearn.tree import DecisionTreeClassifier # Imports the DecisionTreeClassifier class for decision tree classification.
from sklearn.neighbors import KNeighborsClassifier #Imports the KNeighborsClassifier class for k-nearest neighbors classification.
from sklearn.naive_bayes import GaussianNB #  Imports the GaussianNB class for Gaussian Naive Bayes classification.
from sklearn.pipeline import Pipeline #  Imports the Pipeline class for creating a pipeline of data processing steps and a final estimator.
from sklearn.preprocessing import StandardScaler # Imports the StandardScaler class for standardizing feature values.
from sklearn.model_selection import GridSearchCV #Imports the GridSearchCV class for hyperparameter tuning using grid search.
from sklearn.svm import SVC # Imports the SVC class for Support Vector Machine classification.
import seaborn as sns #Imports the Seaborn library for statistical data visualization.  

# %matplotlib inline

"""## Exploratory Analysis

### Read and load Dataset
"""
# ----------------
# loads the dataset from a CSV file, drops the 'id' column, encodes the 'class' column,
# and handles missing values in the 'bare_nucleoli' column by replacing them with the median value.
# --------------
# Load Data
df = pd.read_csv(r'D:\Maitexa\ML\Breast-Cancer-Detection-using-SVM-main\Breast cancer prediction\data.csv')
df.head(10)

#Shape of the Dataset
df.shape

df.describe()

"""## Data pre-processing"""

# Droping the id cloumn which is of no use
df.drop('id', axis=1, inplace=True)

# Columns in the dataset
df.columns



"""## Encoding Categorical Data"""

#Enumerate the diagnosis column such that M = 1, B = 0
df['class'] = df['class'].replace([2,4],[0,1])
df['class'] = df['class'].astype("int64")

#The number of Benign and Maglinant cases from the dataset.
print(df.groupby('class').size())

df.head(10)

df.info()

df[df['bare_nucleoli'] == '?']

df[df['bare_nucleoli'] == '?'].sum()

digits_in_bare_nucleoli = pd.DataFrame(df.bare_nucleoli.str.isdigit())
digits_in_bare_nucleoli

df = df.replace('?', np.nan)
df['bare_nucleoli'].head(24)

df.median()

df = df.fillna(df.median())

df['bare_nucleoli'].head(25)

df.dtypes

df['bare_nucleoli']=df['bare_nucleoli'].astype('int64') #we must do the conversion explicitly

df.dtypes

# get the number of missing data points per column
missing_values_count = df.isnull().sum()
missing_values_count[0:10]# Converting object into Nan and into int value

"""## Data Visualization"""
# ---------------
# Plot histograms for each variable
# This code generates histograms for each variable in the dataset,
# providing a visual representation of the data distribution.
# -----------
sns.set_style('darkgrid')
df.hist(figsize=(30,30), color="Green")
plt.show()

"""## Feature selection"""
# -------
# This section defines a function correlation to identify highly correlated features based on a specified threshold.
# It then calculates and prints the correlated features.
# ---------
plt.figure(figsize=(30,20))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()

#Correlation with output variable
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

#Selecting highly correlated features
corr_features = correlation(df, 0.7)
len(set(corr_features))

corr_features

"""## Train and Test Model"""
#Split the data into predictor variables and target variable, following by breaking them into train and test sets.
# --------------------
# Split the data into training and testing sets 
# --------------------

Y = df['class'].values
X = df.drop('class', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.3, random_state=1)

X_train.shape

X_test.shape

"""## Model Selection
### Baseline algorithm checking

* Analyse and build a model to predict if a given set of symptoms lead to breast cancer. This is a binary classification problem, and a few algorithms are appropriate for use.

* As we do not know which one will perform the best at the point, we will do a quick test on the few appropriate algorithms with default setting to get an early indication of how each of them perform.

* We will use 10 fold cross validation for each testing.

* The following non-linear algorithms will be used, namely:
  * Classification and Regression Trees (CART)
  * Linear Support Vector Machines (SVM)
  * Gaussian Naive Bayes (NB)
  * k-Nearest Neighbors (KNN).
"""

# Testing Options
scoring = 'accuracy'
# Standardize the dataset and reevaluate models
# Define models to train
# --------
# Model evaluation using cross-validation
# This part splits the data into training and 
# testing sets and evaluates the performance of four different machine learning models using 10-fold cross-validation.
# ---------
models= []
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "For %s Model:Mean accuracy is %f (Std accuracy is %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(10,10))
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

"""From the initial run, it looks like CART, SVM, GaussianNB and KNN  performed the best with  above 90% mean accuracy.


## Evaluation of algorithm on Standardised Data

Using pipelines to improve the performance of all machine learning algorithms by using standardised dataset  The improvement is likely for all the models.
"""

# Standardize the dataset
# ------
# Standardize the dataset and reevaluate models
# This section standardizes the data using StandardScaler and reevaluates
# the models, comparing their performance on the standardized data.
# ------
import warnings
pipelines = []

pipelines.append(('Scaled CART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('Scaled SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('Scaled NB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('Scaled KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))

results = []
names = []

kfold = KFold(n_splits= 10)
for name, model in pipelines:
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print( "For %s Model: Mean Accuracy is %f (Std Accuracy is %f)" % (name, cv_results.mean(), cv_results.std()))

fig = plt.figure(figsize=(10,10))
fig.suptitle('Performance Comparison For Standarised Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
# ----------
# Model evaluation on test set
# Finally, the code trains each model on the training set and evaluates its performance on the testing set,
# providing accuracy scores and classification reports.
# ----------

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print("\nModel:",name)
    print("Accuracy score:",accuracy_score(Y_test, predictions))
    print("Classification report:\n",classification_report(Y_test, predictions))

# Accuracy - ratio of correctly predicted observation to the total observations.
# Precision - (false positives) ratio of correctly predicted positive observations to the total predicted positive observations
# Recall (Sensitivity) - (false negatives) ratio of correctly predicted positive observations to the all observations in actual class - yes.
# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false



# ----------------------------------------------------------
# This code aims to guide the process of breast cancer detection through various steps, including data preprocessing, exploration, visualization, feature selection, and model evaluation. The use of cross-validation and standardized data enhances the reliability of the model assessment.