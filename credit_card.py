# Import pandas & sklearn library
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv("cc_approvals.data.csv",header=None)

# Inspect data
data.head()

# To get summary information(Mean, standard deviation, max value, 25th, 50th, and 75th percentiles, and etc) 
data_description = data.describe()
print(data_description)

print("\n")

# To get the data information
data_info = data.info()
print(data_info)

print("\n")

# Inspect the missing values
data.tail(20)

# Replace the ? with NaN
data = data.replace('?',np.NaN)

# Inspect the missing values again
data.tail(20)

# Impute the missing values with mean imputation
data.fillna(data.mean(),inplace=True)

# Count the number of NaNs in the dataset to verify
print(data.isnull().sum())
print(data.tail(20))

# Iterate over each column of cc_apps
for col in data:
    # Check if the column is of object type
    if data[col].dtypes == 'object':
        # Impute with the most frequent value
        data = data.fillna(data[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(data.tail(20))
data.isnull().sum()

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and check if the data type is object type or not. If the data type is object type, the value will be replace with numerical representation
for col in data:
    if data[col].dtypes == 'object':
        data[col] = le.fit_transform(data[col])

#Drop the features 11 and 13 and convert the dataframe to numpy for processing purposes
data = data.drop([11, 13], axis=1)

# to get data values
data = data.values

# Segregate features and labels into separate variables
X,y = data[:,0:12] , data[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Instantiate MinMaxScaler and use it to rescale X_train and X_test to value from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train,y_train)

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", accuracy_score(y_pred,y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_pred,y_test)

# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
print("Best: %f using %s" % (best_score,best_params))






