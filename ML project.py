#!/usr/bin/env python
# coding: utf-8

# In[14]:


# === Common Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# === Scikit-learn Modules ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)


# In[15]:


#Step 1: Data Preparation


df = pd.read_csv("loan_model_ready.csv")

# Encode categorical columns
for col in df.select_dtypes('object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


#Step 2: Model Training + Hyperparameter Tuning

def tune(name, model, params):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_

models = dict([
    tune("Logistic", LogisticRegression(max_iter=1000), {
        'C': [0.1, 1], 'solver': ['liblinear']
    }),
    tune("Tree", DecisionTreeClassifier(), {
        'max_depth': [5, 10], 'criterion': ['gini', 'entropy']
    }),
    tune("Forest", RandomForestClassifier(), {
        'n_estimators': [100, 200], 'max_depth': [10]
    }),
    tune("Boosting", GradientBoostingClassifier(), {
        'n_estimators': [100], 'learning_rate': [0.05, 0.1]
    }),
])


# In[17]:


# Step 3: Model Evaluation

evals = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    evals.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

eval_df = pd.DataFrame(evals)
print(eval_df)


# In[18]:


#Step 4: Dashboard
#🔸 Accuracy Comparison
sns.barplot(x="Model", y="Accuracy", data=eval_df)
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()


# In[19]:


#🔸 Feature Importance

feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models["Boosting"].feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=feat_imp, y='Feature', x='Importance')
plt.title("Feature Importance (Gradient Boosting)")
plt.show()


# In[20]:


#ROC AUC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1])
for name, model in models.items():
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC AUC Comparison")
plt.legend()
plt.show()


# In[21]:


#Training Time Comparison
times = []
for name, model in models.items():
   start = time.time()
   model.fit(X_train, y_train)
   times.append({"Model": name, "Time": time.time() - start})

time_df = pd.DataFrame(times)
sns.barplot(data=time_df, x="Model", y="Time")
plt.title("Training Time Comparison")
plt.show()


# In[ ]:




