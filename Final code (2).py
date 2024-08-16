#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('dataset.csv')
df.head()


# In[3]:


df.shape[0] #Number of instances


# In[4]:


df.shape[1] #Number of features


# In[5]:


df['classification'].value_counts() #Number of instances from each class


# In[6]:


from sklearn.model_selection import train_test_split

X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
display(df) #Load and split the dataset into two parts


# In[7]:


df.isnull().sum() #EDA on the data set


# In[8]:


df.describe() #EDA on the data set


# In[9]:


sns.countplot(x='classification', data=df) #EDA on the data set
plt.show()


# In[10]:


corr_matrix = df.corr(numeric_only=True) #EDA on the data set
sns.heatmap(corr_matrix, annot=True)
plt.show()
df_renamed = df


# In[11]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# In[12]:


# Load your dataset (replace 'dataset.csv' with your actual file)
data = pd.read_csv('dataset.csv')

# Verify that the dataset is loaded correctly
print(data.head())

# Check if 'classification' column exists
if 'classification' in data.columns:
    # Define features (X) and target (y)
    X = data.drop(columns=['classification'])  # Assuming 'classification' is the target column
    y = data['classification']

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {categorical_columns}")

    # Convert categorical data to numeric using one-hot encoding
    encoder = OneHotEncoder(sparse=False, drop='first')
    X_encoded = encoder.fit_transform(X[categorical_columns])

    # Convert the encoded data back to a DataFrame
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical columns and concatenate the encoded columns
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, X_encoded_df], axis=1)

    # Ensure all columns are numeric
    print(X.dtypes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successfully!")


# In[13]:


# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Initialize the Naive Bayes model
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Predict on the test set
y_pred_nb = nb.predict(X_test)

# Evaluate the model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Print accuracy scores
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")

# Optionally, plot confusion matrices
models = {'Logistic Regression': y_pred_log_reg, 'Random Forest': y_pred_rf, 'Naive Bayes': y_pred_nb}

for name, y_pred in models.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[ ]:




