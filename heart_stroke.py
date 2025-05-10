import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('stroke_detection.csv')
df.head()
df.shape
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates()
df.duplicated().sum()
df.describe()
df.info()

#count_plot
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df, palette="Set2")
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='high_blood_pressure', data=df, palette="coolwarm")
plt.title("High Blood Pressure Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.show()

#pie_plot
plt.figure(figsize=(6, 6))
df['gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'pink'])
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(6, 6))
df['at_risk'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'])
plt.title("At-Risk Stroke Patients")
plt.ylabel("")
plt.show()

#bar-plot
plt.figure(figsize=(8, 5))
sns.barplot(x='gender', y='stroke_risk_percentage', data=df, palette="coolwarm")
plt.title("Stroke Risk Percentage by Gender")
plt.xlabel("Gender")
plt.ylabel("Stroke Risk Percentage")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='dizziness', y='stroke_risk_percentage', data=df, palette="muted")
plt.title("Stroke Risk by Dizziness")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.xlabel("Dizziness")
plt.ylabel("Stroke Risk Percentage")
plt.show()

#heat-map
plt.figure(figsize=(10, 6))
sns.heatmap(df[['stroke_risk_percentage', 'high_blood_pressure', 'dizziness']].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(15,6))
sns.histplot(df['age'], kde = True, bins = 10, palette = 'hls')
plt.show()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

fig = px.pie(df, names='gender', title="Gender Distribution", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])

x = df.drop(columns=["at_risk","stroke_risk_percentage"]) 
y = df["at_risk"]
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state=42)
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

new_sample = [float(input(f"Enter {col}: ")) for col in x.columns]
predicted = model.predict([new_sample])
print("Prediction (0=Not at risk, 1=At risk):", predicted[0])
