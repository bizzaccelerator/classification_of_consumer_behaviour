#!/usr/bin/env python
# coding: utf-8

# The first step involves importing the libraries required for the process:
import pandas as pd
import numpy as np

# Model packages used
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# To save the model
import pickle

# Data Exploratory Analysis

# The dataset is loaded as:
users = pd.read_csv("C://Users/jober/Data_Projects/classification_of_consumer_behaviour/Dataset/user_behaviour_dataset.csv", sep=";", )

# Column names in our refined dataframe are converted to lowercase, and spaces are removed for consistency and usability:
users.columns = [name.lower() for name in users.columns]
users.columns = [name.replace(" ","_") for name in users.columns]

# The variables *app_usage_time_(min/day)* and *screen_on_time_(hours/day)* use different time units. To ensure consistency, screen_on_time will be transformed into a minutes-per-day ratio.
users['screen_on_time_(min/day)'] = users['screen_on_time_(hours/day)']*60

# Modify the type of target variable:
users['user_behavior_class'] = users['user_behavior_class'].astype(object)

# The significant variables identifyed were:
significant_var = ['app_usage_time_(min/day)', 'battery_drain_(mah/day)', 'number_of_apps_installed','data_usage_(mb/day)', 
                    'screen_on_time_(min/day)', 'user_behavior_class']

# ## Step 4: Model identification
# The cleaned dataset is filtered to include the significant variables identified above.
users_cleaned = users[significant_var]
users_cleaned.reset_index

# The data is transformed to dictionaries as this example:
users_dict = users.to_dict(orient='records')

# Preparation dataset
X = users_cleaned.drop('user_behavior_class', axis=1)
y = users_cleaned['user_behavior_class']

# Convert target to categorical codes so scikit-learn classification models can work properly
y = y.astype('category').cat.codes

# Turning the dataframes into dictionaries:
X_dic = X.to_dict(orient='records')

# Instanciating the vectorizer for the dictionary:
dv = DictVectorizer(sparse=False)

# Applying the vectorizer:
X_encoded = dv.fit_transform(X_dic)

# Then, dataset is splitted as follows: 60% for training, 20% for validation, and 20% for testing.
# We first split for testing
X_full_train, X_test, y_full_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=2)
# Then we split again for validation
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=2)

# ### - Let's try some models:
# __2. Random Forest Model:__

# The model is trained as follows:
random_forest = RandomForestClassifier(max_depth=2, random_state=2)

# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'n_estimators': [20, 50, 100, 200, 300],    # Number of trees
    'max_depth': [None, 10, 20, 30],            # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],            # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],              # Minimum samples in a leaf
    'max_features': [1.0, 'sqrt', 'log2'],      # Features to consider at each split
    'bootstrap': [True, False],                 # Whether to use bootstrap samples
}

# I'll use GridSearchCV for exhaustive tuning and preventing overfitting.
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

# Generate predictions with the best model 
y_pred = best_rf_model.predict(X_val)

# ## Step 5: Exporting the Model
# The selected model will be exported to a binary file (.bin) for later usage:

# The parameters for the sleected model are:
model_params = best_rf_model.get_params()

# Defining the model name as:
# output_file = f"random_forest_model_estimators={model_params['n_estimators']}_max_features={model_params['max_features']}.bin"
output_file = "test.bin"

# Saving the model for external usage
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,best_rf_model),f_out)