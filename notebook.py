#!/usr/bin/env python
# coding: utf-8

# # Data Exploratory Analysis
# The following outlines the process I used to understand and analyze the dataset.

# In[1]:


# The first step involves importing the libraries required for the process:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The graphics style selected is:
sns.reset_orig() 
plt.style.use('ggplot')

# Statistical packages used
from scipy.stats import shapiro, levene, chi2_contingency, f_oneway, kruskal

# Model packages used
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# To save the model
import pickle


# In[2]:


# The following allows us to view all the columns of the dataset, regardless of its size:
pd.set_option('display.max_columns', None)


# In[3]:


# Then the dataset is loaded as:
users = pd.read_csv("C://Users/jober/Data_Projects/classification_of_consumer_behaviour/Dataset/user_behaviour_dataset.csv", sep=";", )


# ## Step 1: Understanding the data
# This step give us a general sense of the dataset: 

# In[4]:


users.shape


# In[5]:


users.head()


# In[6]:


users.columns


# In[7]:


# Using the info() method, we can quickly identify the data type of each column and detect null values:"
users.info()


# In[8]:


# The describe() function provides basic statistics for the numerical variables in the dataset:
users.describe()


# ## Step 2: Data preparation
# Now that I have a general understanding of the data, some evaluation is needed before proceeding with further analysis.

# In[9]:


# Checking for duplicates:
users.duplicated().sum()


# In[10]:


# Checking for null values 
users.isna().sum()


# In[11]:


# Column names in our refined dataframe are converted to lowercase, and spaces are removed for consistency and usability:
users.columns = [name.lower() for name in users.columns]
users.columns = [name.replace(" ","_") for name in users.columns]


# In[12]:


# The edited dataset is:
users.head()


# The variables *app_usage_time_(min/day)* and *screen_on_time_(hours/day)* use different time units. To ensure consistency, screen_on_time will be transformed into a minutes-per-day ratio.

# In[13]:


users['screen_on_time_(min/day)'] = users['screen_on_time_(hours/day)']*60


# In[14]:


# Modify the type of target variable:
print(users['user_behavior_class'].unique())
users['user_behavior_class'] = users['user_behavior_class'].astype(object)
print('\n While those are numeric values, they are best treated as categories.')


# In[15]:


users.info()


# In[16]:


# The main statistics for out clean dataset are:
users.describe(include='all')


# There are no null values nor repeated rows. This is a clean dataset, then it doesn't requires wrangling. Analysis can now be performed directly.

# In[17]:


users.columns


# In[18]:


# With these sets of variables:
categorical = ['user_id', 'device_model', 'operating_system',
               'age', 'gender', 'user_behavior_class']

numerical = ['app_usage_time_(min/day)', 'screen_on_time_(min/day)', 'battery_drain_(mah/day)', 
             'number_of_apps_installed', 'data_usage_(mb/day)']


# The ranges of al Features are:

# In[19]:


users.drop('user_behavior_class', axis=1).boxplot()
plt.title('Feature Ranges')
plt.tick_params(axis='x', labelrotation=90)
plt.show()


# ## Step 3: Feature understanding
# 
# Now, it is important to understand how the variables behave:
# 
# ### - Target variable (user_behavior_class):

# In[20]:


sns.histplot(
    users,
    x="user_behavior_class", hue="gender",
    multiple="stack",
    edgecolor=".3",
    linewidth=.5,
    log_scale=False,
)
plt.title('Distribution of users by class')
plt.show()


# In[21]:


# The Central Tendency measures are
mean = users['user_behavior_class'].mean()
median = users['user_behavior_class'].median()
print(f"Mean: {mean}, Median: {median}")


# In[22]:


# The Outliers can be identifyed from a Boxplot
sns.boxplot(x=users['user_behavior_class'])
plt.title('Box Plot')
plt.show()


# There are no outliers visible at first glance.
# 
# ### - Analysis for variable "device_model":

# In[23]:


# The variable behave as:
sns.histplot(users, x='device_model', 
             hue='user_behavior_class', 
             multiple="dodge", 
             palette="pastel",
             shrink=.75)
plt.title("Users of different device models")
plt.show()


# In[24]:


users['device_model'].unique()


# In[25]:


# Since both variables are categorical in nature, the chi-squared test of independence will be used. 

# A contingency table is created.
devices = pd.crosstab(users.device_model, users.user_behavior_class)

# Then, I'll apply the non-parametric test for comparaison (Chi-squared):
stat, p, dof, expected = chi2_contingency(devices)
print("\nChi 2 squared Test:")
print(f"X2-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.8995 is higher than 0.05, then there is no variability explained by the type of device used. Therefore, This variable should not be included in the model.

# ### - Analysis for variable "gender":

# In[26]:


# The 'gender' variable behave as:
sns.histplot(x="gender", 
             hue="user_behavior_class", 
             multiple="dodge", 
             data=users, 
             shrink= 0.75, 
             palette="pastel")
plt.title("Distribution of user classes by Gender")
plt.show()


# In[27]:


# Since both variables are categorical in nature, the chi-squared test of independence will be used. 

# A contingency table is created.
genders = pd.crosstab(users.gender, users.user_behavior_class)

# Then, I'll apply the non-parametric test for comparaison (Chi-squared):
stat, p, dof, expected = chi2_contingency(genders)
print("\nChi 2 squared Test:")
print(f"X2-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.1363230 is higher than 0.05, then there is no relationship between gender and user classes. That means, variability in user behaviour explained by the gender of the user. Therefore, This variable should not be included in the model.

# ### - Analysis for variable "operating_system":

# In[28]:


# The 'operating_system' variable behave as:
sns.histplot(x="operating_system", 
             hue="user_behavior_class", 
             multiple="dodge", 
             data=users, 
             shrink= 0.75, 
             palette="pastel")
plt.title("Distribution of user classes by Operating system used")
plt.show()


# In[29]:


# Since both variables are categorical in nature, the chi-squared test of independence will be used. 

# A contingency table is created.
operating_system = pd.crosstab(users.operating_system, users.user_behavior_class)

# Then, I'll apply the non-parametric test for comparaison (Chi-squared):
stat, p, dof, expected = chi2_contingency(operating_system)
print("\nChi 2 squared Test:")
print(f"X2-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.660108 is higher than 0.05, then there is no relationship between operating system used and user classes. That means, variability in user behaviour explained by the operating system of the user's phone. Therefore, This variable should not be included in the model.

# ### - Analysis for variable "age":

# In[30]:


# visualizing the age distribution per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='age', data=users)
plt.title('Age distribution per user category')
plt.show()


# In[31]:


# The firts step is spliting the categories in the column as follows:
groups_age = [group["age"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_age[0]))
print("Class of user 2:", shapiro(groups_age[1]))
print("Class of user 3:", shapiro(groups_age[2]))
print("Class of user 4:", shapiro(groups_age[3]))
print("Class of user 5:", shapiro(groups_age[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_age)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is not homogeinity of variance between group of devices because {p} is higher than 0.05')


# In[32]:


# The assumption of normally distributed data is proven, but there is no evidence of homoscedasticity. 
# Then, I will apply the non-parametric Kruskal-Wallis test to assess differences between groups:
stat, p = kruskal(*groups_age)
print("Kruskal-Wallis H Test:")
print(f"H-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.9939 is far higher than 0.05; then, any observed differences in age between user classes groups are likely due to random variation rather than a true relationship. Therefore, This variable should not be included in the model.

# ### - Analysis for variable "app_usage_time_(min/day)":

# In[33]:


# visualizing the app usage time distribution per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='app_usage_time_(min/day)', data=users)
plt.title('App usage time distribution per user category')
plt.show()


# In[34]:


# The firts step is spliting the categories in the column as follows:
groups_app_usage = [group["app_usage_time_(min/day)"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_app_usage[0]))
print("Class of user 2:", shapiro(groups_app_usage[1]))
print("Class of user 3:", shapiro(groups_app_usage[2]))
print("Class of user 4:", shapiro(groups_app_usage[3]))
print("Class of user 5:", shapiro(groups_app_usage[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_app_usage)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between group of user classes because {p} is smaller than 0.05')


# In[35]:


# The variable 'app_usage_time_(min/day)' follows a normal distribution, and evidence supports homoscedasticity between user groups. 
# Then, I will apply the parametric One-way ANOVA test to assess differences between groups:
stat, p = f_oneway(*groups_app_usage)
print("One-way ANOVA Test:")
print(f"F-statistic: {stat}, P-value: {p}")


# As observed in the image above, some classes have significantly higher usage times than others. The extremely small p-value suggests that the observed differences in group means are almost certainly not due to random chance. Therefore, this variable clearly differentiates user behavior.

# ### - Analysis for variable "screen_on_time_(min/day)":

# In[36]:


# visualizing the screen exposition time per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='screen_on_time_(min/day)', data=users)
plt.title('Screen on time distribution per user category')
plt.show()


# In[37]:


# The firts step is spliting the categories in the column as follows:
groups_screen = [group["screen_on_time_(min/day)"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_screen[0]))
print("Class of user 2:", shapiro(groups_screen[1]))
print("Class of user 3:", shapiro(groups_screen[2]))
print("Class of user 4:", shapiro(groups_screen[3]))
print("Class of user 5:", shapiro(groups_screen[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_screen)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between group of user classes because {p} is smaller than 0.05')


# In[38]:


# The variable 'screen_on_time_(min/day)' follows a normal distribution, and evidence supports homoscedasticity between user groups. 
# Then, I will apply the parametric One-way ANOVA test to assess differences between groups:
stat, p = f_oneway(*groups_screen)
print("One-way ANOVA Test:")
print(f"F-statistic: {stat}, P-value: {p}")


# Based on the image above, certain classes exhibit notably higher usage times compared to others. The very small p-value indicates that the differences in group means are highly unlikely to be due to random variation. As a result, this variable effectively distinguishes user behavior.

# ### - Analysis for variable "battery_drain_(mah/day)":

# In[39]:


# visualizing the battery consumption per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='battery_drain_(mah/day)', data=users)
plt.title('Battery consumption per user category')
plt.show()


# In[40]:


# The firts step is spliting the categories in the column as follows:
groups_battery = [group["battery_drain_(mah/day)"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_battery[0]))
print("Class of user 2:", shapiro(groups_battery[1]))
print("Class of user 3:", shapiro(groups_battery[2]))
print("Class of user 4:", shapiro(groups_battery[3]))
print("Class of user 5:", shapiro(groups_battery[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_battery)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between group of user classes because {p} is smaller than 0.05')


# In[41]:


# The variable 'battery_drain_(mah/day)' follows a normal distribution, and evidence supports homoscedasticity between user groups. 
# Then, I will apply the parametric One-way ANOVA test to assess differences between groups:
stat, p = f_oneway(*groups_battery)
print("One-way ANOVA Test:")
print(f"F-statistic: {stat}, P-value: {p}")


# The P-value calculated is smaller than 1e300. Hence, it's smaller than 0.05, then there is variability explained in user classes by the battery consumption patterns. Therefore, This variable should be included in the model.

# ### - Analysis for variable "number_of_apps_installed":

# In[42]:


# visualizing the number of apps installed per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='number_of_apps_installed', data=users)
plt.title('Number of apps installed per user category')
plt.show()


# In[43]:


# The firts step is spliting the categories in the column as follows:
groups_app_number = [group["number_of_apps_installed"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_app_number[0]))
print("Class of user 2:", shapiro(groups_app_number[1]))
print("Class of user 3:", shapiro(groups_app_number[2]))
print("Class of user 4:", shapiro(groups_app_number[3]))
print("Class of user 5:", shapiro(groups_app_number[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_app_number)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between group of user classes because {p} is smaller than 0.05')


# In[44]:


# The variable 'number_of_apps_installed' follows a normal distribution, and evidence supports homoscedasticity between user groups. 
# Then, I will apply the parametric One-way ANOVA test to assess differences between groups:
stat, p = f_oneway(*groups_app_number)
print("One-way ANOVA Test:")
print(f"F-statistic: {stat}, P-value: {p}")


# The P-value calculated is smaller than 0.05, then there is variability explained by the number of apps installed by the different users. Therefore, This variable should be included in the model.

# ### - Analysis for variable "data_usage_(mb/day)":

# In[45]:


# visualizing the data usage distribution per user category
plt.figure(figsize=(10,6))
sns.boxplot(x='user_behavior_class', y='data_usage_(mb/day)', data=users)
plt.title('Data usage distribution per user category')
plt.show()


# In[46]:


# The firts step is spliting the categories in the column as follows:
groups_data = [group["data_usage_(mb/day)"].values for name, group in users.groupby("user_behavior_class")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Class of user 1:", shapiro(groups_data[0]))
print("Class of user 2:", shapiro(groups_data[1]))
print("Class of user 3:", shapiro(groups_data[2]))
print("Class of user 4:", shapiro(groups_data[3]))
print("Class of user 5:", shapiro(groups_data[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_data)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between group of user classes because {p} is smaller than 0.05')


# In[47]:


# The variable 'data_usage_(mb/day)' follows a normal distribution, and evidence supports homoscedasticity between user groups. 
# Then, I will apply the parametric One-way ANOVA test to assess differences between groups:
stat, p = f_oneway(*groups_data)
print("One-way ANOVA Test:")
print(f"F-statistic: {stat}, P-value: {p}")


# The calculated p-value is less than 0.05, indicating that the amount of data used by different users explains some of the variability. Consequently, this variable should be included in the model.

# ### - Analysis of information explaind by categorical variables:
# 
# To determine the key variables, I use the following function: 

# In[48]:


def mutual_info_categorical(series):
    return mutual_info_score(series, users.user_behavior_class)


# Then, applying the function above I got: 

# In[49]:


mi = users[categorical].apply(mutual_info_categorical)
mi.sort_values(ascending=False)


# From this, it becomes clear that these variables are not significant for our future elaborated model.

# ### - Analysis of correlated variables:

# In[50]:


users[numerical].corrwith(users.user_behavior_class)


# From the results above, it is evident that all numerical variables in our dataset are positively correlated with the target variable (user_behavior_class). This indicates that higher user categories require more resources from their phones.

# Based on the analysis developed, the variables useful for our purpose of classifying users are:

# In[51]:


# The significant variables identifyed were:
significant_var = ['app_usage_time_(min/day)', 'battery_drain_(mah/day)', 'number_of_apps_installed','data_usage_(mb/day)', 
                    'screen_on_time_(min/day)', 'user_behavior_class']

# The not significant variables identifyed were:
not_significant_var = ['device_model', 'gender', 'operating_system', 'age']


# ## Step 4: Model identification
# The cleaned dataset is filtered to include the significant variables identified above.

# In[52]:


users_cleaned = users[significant_var]
users_cleaned.reset_index
users_cleaned.head()


# In[53]:


# The data is transformed to dictionaries as this example:
users_dict = users.to_dict(orient='records')
users_dict = users_dict[3]
users_dict


# Working dataset is prepared and splitted as follows:

# In[54]:


# Preparation dataset
X = users_cleaned.drop('user_behavior_class', axis=1)
y = users_cleaned['user_behavior_class']

# Convert target to categorical codes so scikit-learn classification models can work properly
y = y.astype('category').cat.codes

# Turning the dataframes into dictionaries:
X_dic = X.to_dict(orient='records')


# In[55]:


# Instanciating the vectorizer for the dictionary:
dv = DictVectorizer(sparse=False)

# Applying the vectorizer:
X_encoded = dv.fit_transform(X_dic)


# In[56]:


# The vectorized rows are transformed to the form of:
print(f'The column names are: {dv.get_feature_names_out()}')
print('\n The first element of the transformed dataset is: ')
X_encoded[0]


# Then, dataset is splitted as follows: 60% for training, 20% for validation, and 20% for testing.

# In[57]:


# We first split for testing
X_full_train, X_test, y_full_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=2)

# Then we split again for validation
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=2)

# The lenght of the datasets can be validated as:
print(f'The number of registries in the train dataset is {len(X_train)}, in the validation dataset is {len(X_val)}, and in the test dataset is {len(X_test)}.')


# ### - Let's try some models:
# 
# __1. Decision Tree Model:__

# In[58]:


# The model is trained as  follows:
decision_tree = DecisionTreeClassifier(max_depth=2)

# The trained model is used to predict the values in the test dataset:
decision_tree.fit(X_train, y_train)
y_pred_val = decision_tree.predict(X_val)

# The indicator chosen for assessing the validity of the model is Accuracy:
print("Decision Tree Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))


# In[59]:


# The parameters of the trained model are:
decision_tree.get_params()


# In[60]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {    
    'max_depth': [None, 3, 5, 10, 20, 30],              # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],                    # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],                      # Minimum samples in a leaf
    'max_features': [None, 'sqrt', 'log2']              # Features to consider at each split
}


# In[61]:


# I'll use GridSearchCV for exhaustive tuning and preventing overfitting.
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')

grid_search.fit(X_train, y_train)

print("Best parameters for my Decision Tree model are:", grid_search.best_params_)
best_dt_model = grid_search.best_estimator_


# In[62]:


# Generate predictions with the best model 
y_pred_val_opt = best_dt_model.predict(X_val)

# Create a confusion matrix 
cm_dt = confusion_matrix(y_val, y_pred_val_opt)


# In[63]:


# Display the confusion matrix created
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=best_dt_model.classes_).plot()
plt.title('Confusion Matrix for the optimzed Decision Tree Model')
plt.show()


# In[64]:


# The classification_report provides a comprehensive overview of classification performance across 
# multiple metrics (precision, recall, f1-score) for each class in the model
dt_report = classification_report(y_val, y_pred_val_opt, target_names=['1', '2', '3', '4', '5'])
print(dt_report)


# __2. Random Forest Model:__

# In[65]:


# The model is trained as follows:
random_forest = RandomForestClassifier(max_depth=2, random_state=2)

# The trained model is used to predict the values in the test dataset:
random_forest.fit(X_train, y_train)
y_pred_val = random_forest.predict(X_val)

# The indicator chosen for assessing the validity of the model is Accuracy:
print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))


# In[66]:


# The parameters of the trained model are:
random_forest.get_params()


# In[67]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'n_estimators': [20, 50, 100, 200, 300],    # Number of trees
    'max_depth': [None, 10, 20, 30],            # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],            # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],              # Minimum samples in a leaf
    'max_features': [1.0, 'sqrt', 'log2'],      # Features to consider at each split
    'bootstrap': [True, False],                 # Whether to use bootstrap samples
}


# In[68]:


# I'll use GridSearchCV for exhaustive tuning and preventing overfitting.
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')
grid_search.fit(X_train, y_train)

print("Best parameters for our Random forest model are:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_


# In[69]:


# Generate predictions with the best model 
y_pred_val_opt = best_rf_model.predict(X_val)

# Create a confusion matrix 
cm_rf = confusion_matrix(y_val, y_pred_val_opt)


# In[70]:


# Display the confusion matrix created
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=best_rf_model.classes_).plot()
plt.title('Confusion Matrix for the optimzed Random Forest Model')
plt.show()


# In[71]:


# The classification_report provides a comprehensive overview of classification performance across 
# multiple metrics (precision, recall, f1-score) for each class in the model
rf_report = classification_report(y_val, y_pred_val_opt, target_names=['1', '2', '3', '4', '5'])
print(rf_report)


# __3. Gradient Boosted Trees (GBT) Model:__

# In[72]:


# The model is trained as follows:
gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=2)

# The trained model is used to predict the values in the test dataset:
gbt.fit(X_train, y_train)
y_pred_val = gbt.predict(X_val)

# The indicator chosen for assessing the validity of the model is Accuracy:
print("Gradient Boosted Trees (GBT) Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))


# In[73]:


gbt.get_params()


# In[74]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[75]:


# I'll use GridSearchCV for exhaustive tuning and preventing overfitting.
grid_search = GridSearchCV(estimator=gbt, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')
grid_search.fit(X_train, y_train)

print("Best parameters for our Random forest model are:", grid_search.best_params_)
best_gbt_model = grid_search.best_estimator_


# In[76]:


# Generate predictions with the best model 
y_pred_val_opt = best_gbt_model.predict(X_val)

# Create a confusion matrix 
cm_gbt = confusion_matrix(y_val, y_pred_val_opt)


# In[77]:


# Display the confusion matrix created
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gbt, display_labels=best_gbt_model.classes_).plot()
plt.title('Confusion Matrix for the optimzed GBT Model')
plt.show()


# In[78]:


# The classification_report provides a comprehensive overview of classification performance across 
# multiple metrics (precision, recall, f1-score) for each class in the model
gbt_report = classification_report(y_val, y_pred_val_opt, target_names=['1', '2', '3', '4', '5'])
print(gbt_report)


# __4. Support Vector Classifier Model:__

# SVM relies on distance and kernel computations, that's why SMV is highly sensitive to feature magnitudes.

# In[79]:


# Standardize features for this model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)


# In[80]:


# The model is trained as follows:
svm_class = SVC(kernel='rbf', gamma='auto', C=1.0, random_state=2)

# The trained model is used to predict the values in the test dataset:
svm_class.fit(X_train_scaled, y_train)
y_pred_val = svm_class.predict(X_val_scaled)

# The indicator chosen for assessing the validity of the model is Accuracy:
print("Support Vector Machine (SVM) Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))


# In[81]:


# The parameters of the SVM are:
svm_class.get_params()


# In[82]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}


# In[83]:


# I'll use GridSearchCV for exhaustive tuning and preventing overfitting.
grid_search = GridSearchCV(estimator=svm_class, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')
grid_search.fit(X_train_scaled, y_train)

print("Best parameters for our Support Vector Machime (SVM) model are:", grid_search.best_params_)
best_svm_model = grid_search.best_estimator_


# In[84]:


# Generate predictions with the best model 
y_pred_val_opt = best_svm_model.predict(X_val_scaled)

# Create a confusion matrix 
cm_svm = confusion_matrix(y_val, y_pred_val_opt)


# In[85]:


# Display the confusion matrix created
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=best_svm_model.classes_).plot()
plt.title('Confusion Matrix for the optimzed SVM Model')
plt.show()


# In[86]:


# The classification_report provides a comprehensive overview of classification performance across 
# multiple metrics (precision, recall, f1-score) for each class in the model
svm_report = classification_report(y_val, y_pred_val_opt, target_names=['1', '2', '3', '4', '5'])
print(svm_report)


# __5. Neural Network Model:__

# __To summarize, the chosen models produce the following Accuracy, Precision and Recall scores when applied to the test dataset:__

# In[87]:


# The list of models evaluated are:
listed_models = { 
                 "Raw Decision Tree": decision_tree, 
                 "Optimized Decision Tree": best_dt_model, 
                  "Raw Random forest": random_forest,
                  "Optimized Random Forest": best_rf_model,
                  "Raw Gradient Boosted trees": gbt, 
                  "Optimized GBT": best_gbt_model,
                  "Raw SVM Classifier": svm_class, 
                  "Optimized SVM Classifier": best_svm_model,
                }


# In[88]:


# The evaluation is performed by: 
result_scores = [] 
for name, model in listed_models.items():
    if 'SVM' in name:
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    #Then the calculation of metris is:
    accuracy_model = accuracy_score(y_test, y_pred)
    precision_model = precision_score(y_test, y_pred, average='weighted')
    recall_model = recall_score(y_test, y_pred, average='weighted')
    
    result_scores.append([name, accuracy_model, precision_model, recall_model])

scores_summary = pd.DataFrame(result_scores, columns=['Model', 'Accuracy', 'Precision', 'Recall'])

# The summary of evaluation metrics is:
scores_summary


# __Conclusions:__
# 
# The best model is the Optimized Gradient Boosted Trees (Grid GBT) because it produces the lowest average deviation from the test values (41.775) and provides the highest explanation of variability in yield production (90.1378%).
