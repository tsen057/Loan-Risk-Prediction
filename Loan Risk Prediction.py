#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('C:/Users/tejas/OneDrive/Desktop/Pet Projects/loan risk prediction/Loan_default.csv')

# Show the first few rows of the dataset
print(df.head())


# In[4]:


# Check for missing values
print(df.isnull().sum())

# Handle missing values - filling missing numerical data with mean, categorical with mode
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Income'].fillna(df['Income'].mean(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['CreditScore'].fillna(df['CreditScore'].mean(), inplace=True)
df['MonthsEmployed'].fillna(df['MonthsEmployed'].mean(), inplace=True)
df['NumCreditLines'].fillna(df['NumCreditLines'].mean(), inplace=True)
df['InterestRate'].fillna(df['InterestRate'].mean(), inplace=True)
df['LoanTerm'].fillna(df['LoanTerm'].mode()[0], inplace=True)
df['DTIRatio'].fillna(df['DTIRatio'].mean(), inplace=True)
df['Education'].fillna(df['Education'].mode()[0], inplace=True)
df['EmploymentType'].fillna(df['EmploymentType'].mode()[0], inplace=True)
df['MaritalStatus'].fillna(df['MaritalStatus'].mode()[0], inplace=True)
df['HasMortgage'].fillna(df['HasMortgage'].mode()[0], inplace=True)
df['HasDependents'].fillna(df['HasDependents'].mode()[0], inplace=True)
df['LoanPurpose'].fillna(df['LoanPurpose'].mode()[0], inplace=True)
df['HasCoSigner'].fillna(df['HasCoSigner'].mode()[0], inplace=True)
df['Default'].fillna(df['Default'].mode()[0], inplace=True)

# Convert categorical variables to numerical using LabelEncoder
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['EmploymentType'] = label_encoder.fit_transform(df['EmploymentType'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['HasMortgage'] = df['HasMortgage'].apply(lambda x: 1 if x == 'Yes' else 0)
df['HasDependents'] = df['HasDependents'].apply(lambda x: 1 if x == 'Yes' else 0)
df['LoanPurpose'] = label_encoder.fit_transform(df['LoanPurpose'])
df['HasCoSigner'] = df['HasCoSigner'].apply(lambda x: 1 if x == 'Yes' else 0)

# Display the cleaned data
print(df.head())


# In[5]:


# Select features (independent variables) and target (dependent variable)
features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 
            'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 'EmploymentType', 
            'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
X = df[features]
y = df['Default']


# In[6]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Build a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the loan default status on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Display feature importance
features_importance = model.feature_importances_
sns.barplot(x=features, y=features_importance)
plt.title("Feature Importance")
plt.xticks(rotation=90)
plt.show()


# In[8]:


# Add the predicted risk to the original dataset
df['Predicted_Default'] = model.predict(scaler.transform(df[features]))

df.head()

# Save to CSV for Power BI import
df.to_csv('loan_risk_predictions.csv', index=False)


# In[9]:


df.head()


# In[ ]:




