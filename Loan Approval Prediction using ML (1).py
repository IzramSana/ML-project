#!/usr/bin/env python
# coding: utf-8

# In[23]:


##  LOAD DATASET 
import pandas as pd

df = pd.read_csv("loan_model_ready.csv")
df.head()


# In[24]:


# STEP 1: DATA EXPLORATION(EDA)

import pandas as pd

# Load dataset
df = pd.read_csv("loan_model_ready.csv")

# Show first 5 rows
print("🔹 First 5 rows:")
display(df.head())

# Dataset shape (rows, columns)
print("\n🔹 Dataset shape:")
print(df.shape)

# Column data types and null values
print("\n🔹 Data types and null values:")
print(df.info())

# Summary statistics for numeric columns
print("\n🔹 Summary statistics:")
display(df.describe())

# Count of missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# List of all columns
print("\n🔹 All column names:")
print(df.columns.tolist())

# Count values of target column (Update this if your target is different)
print("\n🔹 Target column value counts:")
print(df['loan_status'].value_counts())  


# In[25]:


# 📌 Splitting the dataset into training and testing sets (80/20 split)
# Features (X) exclude the target column 'loan_status'
# Target (y) is the 'loan_status' column
# This step is essential for evaluating model performance on unseen data


from sklearn.model_selection import train_test_split

# Step 1: Define features and target
X = df.drop("loan_status", axis=1)  # drop target column from features
y = df["loan_status"]               # target column

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set:", X_train.shape)
print("Test Set:", X_test.shape)


# In[26]:


# Label Encoding: Convert all categorical columns into numeric values using LabelEncoder

from sklearn.preprocessing import LabelEncoder

# Copy original DataFrame (assuming it's called df)
df_encoded = df.copy()

# Encode all object (string/categorical) columns
label_encoders = {}
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

df_encoded.head()


# In[27]:


# Split data into training and testing sets


X = df_encoded.drop("loan_status", axis=1)
y = df_encoded["loan_status"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


#STEP 2: TRAIN AND COMPARE MODLES
# Train and evaluate multiple classification models (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting)
# Compare their performance using Accuracy, Precision, Recall, and F1-Score
# Visualize accuracy scores for comparison


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Step 2: Train & Evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    })

# Step 3: Convert to DataFrame
results_df = pd.DataFrame(results)

# Step 4: Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Show all metrics
results_df


# In[29]:


# STEP 3: HYPERPARAMETER TUNING
# Hyperparameter tuning for Decision Tree using GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Decision Tree Hyperparameter Tuning
dt_params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)

print("Best Parameters for Decision Tree:", dt_grid.best_params_)
print("Best Accuracy Score:", dt_grid.best_score_)


# In[30]:


# Random Forest Hyperparameter Tuning

from sklearn.ensemble import RandomForestClassifier

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("Best Parameters for Random Forest:", rf_grid.best_params_)
print("Best Accuracy Score:", rf_grid.best_score_)


# In[31]:


# logistic regression Hyperparameter tuning

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the model
log_reg = LogisticRegression(max_iter=1000)

# Define the parameter grid
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

# Perform GridSearchCV
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters for Logistic Regression:", grid_lr.best_params_)
print("Best Accuracy for Logistic Regression:", grid_lr.best_score_)


# In[32]:


# Gradient boosting 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Model
gb = GradientBoostingClassifier()

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# Grid Search
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid,
                               cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the model
grid_search_gb.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters for Gradient Boosting:", grid_search_gb.best_params_)
print("Best Accuracy Score:", grid_search_gb.best_score_)

# Predict and evaluate
from sklearn.metrics import accuracy_score, classification_report

y_pred_gb = grid_search_gb.predict(X_test)
print("Gradient Boosting Test Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))


# In[33]:


# STEP 4: Model Comparison & Visualization
# Evaluate all tuned models on the test set using accuracy, precision, recall, and F1-score
# Store the results in a DataFrame for easy comparison

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Tuned models
best_models = {
    "Logistic Regression": grid_lr.best_estimator_,
    "Decision Tree": dt_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
    "Gradient Boosting": grid_search_gb.best_estimator_
}

# Evaluation results
eval_results = []

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    eval_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

# Convert to DataFrame
import pandas as pd
eval_df = pd.DataFrame(eval_results)
print(eval_df)


# In[34]:


# STEP 5: CREATING DASHBOARDS 

# 📊 Visualize and compare the performance metrics (Accuracy, Precision, Recall, F1-Score) of all tuned models using a bar plot

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Melt the DataFrame for easier plotting with seaborn
eval_df_melted = pd.melt(eval_df, id_vars=["Model"], 
                         value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                         var_name="Metric", value_name="Score")

# Plot
sns.barplot(data=eval_df_melted, x="Model", y="Score", hue="Metric")
plt.title(" Performance Comparison of Tuned Models", fontsize=16)
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[35]:


# 📊 Plotting a bar chart to visually compare the performance metrics (Accuracy, Precision, Recall, F1-Score) of all tuned models

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Melt the DataFrame for easier plotting
eval_melted = eval_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plot
sns.barplot(data=eval_melted, x="Model", y="Score", hue="Metric")
plt.title("Performance Comparison of Tuned Models")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[36]:


# Plot confusion matrix for each tuned model to visualize prediction accuracy across classes

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix for each model
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    print(f"\n🔹 Confusion Matrix for {name}")
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()


# In[37]:


# Show bar chart comparing model performance using all metrics

import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style="whitegrid")

# Melt the DataFrame to long format for seaborn
eval_melted = eval_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=eval_melted, x="Model", y="Score", hue="Metric", palette="Set2")

plt.title(" Model Performance Comparison", fontsize=16)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[38]:


# 🔍 Visualize the importance of each feature using the best Gradient Boosting model
# Helps identify which features contribute most to loan approval prediction

import matplotlib.pyplot as plt
import seaborn as sns

# Using the best gradient boosting model for feature importance
feature_importances = grid_search_gb.best_estimator_.feature_importances_
features = X.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Gradient Boosting)')
plt.tight_layout()
plt.show()


# In[39]:


# Plot ROC AUC curves for each model to compare classification performance using true positive and false positive rates

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize the target for ROC AUC (works if binary classification)
y_test_bin = label_binarize(y_test, classes=[0, 1])  # Adjust classes if needed

plt.figure(figsize=(10, 6))

for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for models that don't support predict_proba (like some SVMs)
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[40]:


# ⏱️ Compare the training time of each tuned model to assess computational efficiency

import time

train_times = []

for name, model in best_models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_times.append({'Model': name, 'Training Time (s)': end - start})

train_time_df = pd.DataFrame(train_times)

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Training Time (s)', data=train_time_df)
plt.title('Model Training Time Comparison')
plt.tight_layout()
plt.show()

train_time_df


# 📌 Conclusion and Key Insights
# 
# After training and tuning multiple classification models on the loan dataset, we observed the following:
# 
# 1.Best Model:
# Logistic Regression performed the best with an accuracy of 0.79, followed closely by Gradient Boosting and Random Forest (0.77 each).
# 
# 2.Effect of Hyperparameter Tuning:
# Tuning significantly boosted performance, especially for Decision Tree and Gradient Boosting models.
# 
# 3.Training Time:
# Logistic Regression had the fastest training time, while ensemble models like Random Forest and Gradient Boosting took longer.
# 
# 4.Important Features:
# Features like person_income, loan_amnt, and loan_int_rate were most influential in predicting loan approval.

# In[ ]:




