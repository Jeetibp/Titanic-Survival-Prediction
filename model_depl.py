#!/usr/bin/env python
# coding: utf-8

# ## 📊 Data Science Analysis of Titanic Dataset
# 
# ### 1️⃣ Introduction
# - **Objective**: Analyze the Titanic dataset and build a predictive model for passenger survival.
# - **Dataset**: Contains features like `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.
# 
# ### 2️⃣ Exploratory Data Analysis (EDA)
# #### 🔍 Dataset Overview
# - Displayed dataset information using:
#   - `train_df.info()` and `test_df.info()` to show column types and non-null counts.
#   - `train_df.describe()` for summary statistics of numerical features.
# - Visualized data distributions and relationships:
#   - Histograms for numerical features like `Age` and `Fare`.
#   - Bar plots for categorical features like `Sex` and `Pclass` vs. `Survived`.
# - Correlation matrix heatmap to identify feature relationships.
# 
# ### 3️⃣ Data Preprocessing
# #### 🧹 Handling Missing Values
# - Filled missing `Age` values with median.
# - Filled missing `Embarked` values with mode.
# - Filled missing `Fare` values in test set with median.
# - Dropped `Cabin` column due to high percentage of missing values.
# 
# #### 🔢 Feature Engineering
# - Created `FamilySize` feature by combining `SibSp` and `Parch`.
# - Encoded `Sex` as numeric (0 for male, 1 for female).
# - One-hot encoded `Embarked` feature.
# 
# #### 📉 Feature Selection
# - Dropped less relevant features like `Name`, and `Ticket`.
# 
# ### 4️⃣ Model Building
# - Split data into features (X) and target (y) for training set.
# - Scaled numerical features using `StandardScaler`.
# - Built and trained a logistic regression model.
# 
# ### 5️⃣ Model Evaluation
# - Calculated performance metrics on training set:
#   - Accuracy
#   - Precision
#   - Recall
#   - F1 Score
#   - ROC AUC Score
# - Visualized confusion matrix for model performance.
# 
# ### 6️⃣ Predictions and Submission
# - Made predictions on the test set.
# - Created a submission file with `PassengerId` and predicted `Survived` values.
# 
# ### 7️⃣ Visualization of Results
# - Created bar plots and scatter plots to visualize predictions:
#   - Bar chart showing count of survived vs. not survived predictions.
#   - Scatter plot of `PassengerId` vs. predicted survival.
# 
# ### 🔚 Conclusion
# - Logistic regression model achieved good performance in predicting survival.
# - Key features influencing survival include `Sex`, `Pclass`, and `Fare`.
# - The model can be further improved by trying other algorithms or ensemble methods.

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[3]:


#Load both datasets
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')
print("Step 1: Datasets loaded successfully")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")


# In[4]:


#Check for null values in both datasets
print("\nStep 2: Checking for null values")
print("Train dataset null values:")
print(train_df.isnull().sum())
print("\nTest dataset null values:")
print(test_df.isnull().sum())


# In[5]:


# Step 3: Dataset information and summary statistics
print("\nStep 3: Dataset information")
print("Train dataset info:")
train_df.info()
print("\nTest dataset info:")
test_df.info()


# In[6]:


print("\nSummary statistics for Train dataset:")
print(train_df.describe())


# In[7]:


#Handle missing values (Avoiding SettingWithCopyWarning)
print("\nStep 4: Handling missing values")

# Train dataset
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())  # Fill missing 'Age' with median
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])  # Fill missing 'Embarked' with mode
train_df.drop(columns=['Cabin'], inplace=True)  # Drop 'Cabin' due to a high percentage of missing values

# Test dataset
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())  # Fill missing 'Age' with median
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())  # Fill missing 'Fare' with median
test_df.drop(columns=['Cabin'], inplace=True)  # Drop 'Cabin' due to a high percentage of missing values


# In[8]:


print("Missing values after handling:")
print("Train dataset:")
print(train_df.isnull().sum())
print("\nTest dataset:")
print(test_df.isnull().sum())


# In[9]:


#Convert categorical variables to numeric (One-hot encoding)
print("\nStep 5: Converting categorical variables to numeric")
for df_name, df in zip(["Train", "Test"], [train_df, test_df]):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numeric (0 = male, 1 = female)
    encoded_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)  # One-hot encode 'Embarked'
    df.drop(columns=['Embarked'], inplace=True)  # Drop original 'Embarked' column after encoding
    df[encoded_embarked.columns] = encoded_embarked  # Add encoded columns back to the DataFrame
    
    print(f"\n{df_name} dataset after encoding:")
    print(df.head())

# Debugging print to ensure no leftover non-numeric columns exist before correlation heatmap
print("\nColumns in Train Dataset after encoding and before correlation heatmap:")
print(train_df.dtypes)


# In[10]:


# Histograms for numerical features
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.histplot(train_df['Age'].dropna(), kde=True)
plt.title('Distribution of Age')
plt.subplot(122)
sns.histplot(train_df['Fare'].dropna(), kde=True)
plt.title('Distribution of Fare')
plt.tight_layout()
plt.show()


# In[11]:


# Bar plots for categorical features
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Survival Count by Sex')
plt.subplot(122)
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival Count by Passenger Class')
plt.tight_layout()
plt.show()


# In[12]:


#Correlation heatmap before removing unnecessary columns
columns_to_drop_for_corr = ['Name', 'Ticket']
train_corr_df = train_df.drop(columns=columns_to_drop_for_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(train_corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Train Dataset")
plt.show()


# In[13]:


# Feature Engineering: Create FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1


# In[14]:


# Drop weakly correlated features based on heatmap analysis
features_to_drop = ['Age', 'SibSp', 'Parch']
train_df.drop(columns=features_to_drop, inplace=True)
test_df.drop(columns=features_to_drop, inplace=True)


# In[15]:


# Debugging print to ensure no leftover non-numeric columns exist before modeling
print("\nColumns in Train Dataset after feature engineering:")
print(train_df.dtypes)


# In[16]:


#Prepare data for modeling (Feature Selection)
selected_features = ['Sex', 'Pclass', 'Fare', 'Embarked_Q', 'Embarked_S', 'FamilySize']
X_train = train_df[selected_features]
y_train = train_df['Survived']
X_test = test_df[selected_features]


# In[17]:


#Scale numerical features (optional for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[18]:


# Train logistic regression model
model = LogisticRegression(random_state=100)
model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained")


# In[19]:


#Evaluate model performance on the training set (optional step)
y_train_pred = model.predict(X_train_scaled)
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)
roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])


# In[20]:


print("Model evaluation metrics on training set")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


# In[21]:


# Confusion matrix visualization (optional)
conf_matrix = confusion_matrix(y_train, y_train_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[22]:


#Make predictions on the test set
y_test_pred = model.predict(X_test_scaled)


# In[23]:


#Create a submission file with predictions for test data
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_test_pred})
submission.to_csv('titanic_submission.csv', index=False)
print("\nStep 11: Submission file created as 'titanic_submission.csv'")


# ### 🔚 Conclusion
# - Logistic regression model achieved good performance in predicting survival.
# - Key features influencing survival include `Sex`, `Pclass`, and `Fare`.
# - The model can be further improved by trying other algorithms or ensemble methods.

# In[25]:


# Import necessary libraries
import pandas as pd
import plotly.express as px


# In[26]:


# Load the submission file
submission = pd.read_csv('titanic_submission.csv')
print("\nSubmission File Preview:")
print(submission.head())


# In[27]:


# Interactive Bar Chart: Count of Survived (0 vs 1)
survival_counts = submission['Survived'].value_counts().reset_index()
survival_counts.columns = ['Survived', 'Count']  # Rename columns for clarity

fig = px.bar(
    survival_counts,
    x='Survived',
    y='Count',
    labels={'Survived': 'Survival Status', 'Count': 'Number of Passengers'},
    title='Survival Count (0 = Did Not Survive, 1 = Survived)',
    text='Count'
)
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(showlegend=False)
fig.show()


# In[28]:


# Interactive Scatter Plot: PassengerId vs Survived
fig2 = px.scatter(
    submission,
    x='PassengerId',
    y='Survived',
    color='Survived',
    labels={'PassengerId': 'Passenger ID', 'Survived': 'Survival Status'},
    title='Passenger Survival Status (Interactive Scatter Plot)',
)
fig2.show()


# **Interview questions**

# **What is the difference between precision and recall?**
# 
# 🎯 Precision measures the accuracy of positive predictions made by a model, while 🔍 recall measures the completeness in identifying all relevant instances.
# 
# Precision is calculated as: TP / (TP + FP)
# Where TP = True Positives, FP = False Positives
# 
# Recall is calculated as: TP / (TP + FN)
# Where TP = True Positives, FN = False Negatives
# 
# 🎯 Precision focuses on the proportion of correct positive predictions.
# 
# 🔍 Recall indicates the percentage of actual positives that were identified.

# **What is cross-validation, and why is it important in binary classification?**
# 
# 🔀 Cross-validation is a statistical technique used to assess how well a machine learning model will generalize to an independent dataset. It involves partitioning the data into subsets, training the model on some subsets, and validating it on others.
# 
# Cross-validation is important in binary classification for several reasons:
# 
# 📊 It provides a more robust estimate of model performance by using multiple train-test splits.
# 🛡️ It helps prevent overfitting by ensuring the model performs well on different subsets of data.
# 🔧 It allows for better tuning of hyperparameters.
# 📈 It gives a clearer measure of how the model will perform on unseen data.
# 
# Common methods include k-fold cross-validation, where the data is split into k subsets, and the model is trained and tested k times.

# In[32]:


import pickle

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# In[ ]:




