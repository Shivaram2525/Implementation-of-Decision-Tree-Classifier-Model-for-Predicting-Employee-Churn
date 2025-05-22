# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.

2. Find the null values and count them.

3. Count number of left values.

4. From sklearn import LabelEncoder to convert string values to numerical values.

5. From sklearn.model_selection import train_test_split.

6. Assign the train dataset and test dataset.

7. From sklearn.tree import DecisionTreeClassifier.

8. Use criteria as entropy.

9. From sklearn import metrics.

10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shivaram M.
RegisterNumber: 212223040195
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
df=pd.read_csv('Employee Data.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
le=LabelEncoder()
df['Departments '].unique()
df['Departments ']=le.fit_transform(df['Departments '])
df['Departments ']
df['salary'].unique()
category=['low','medium','high']
oe=OrdinalEncoder(categories=[category])
df['salary']=oe.fit_transform(df[['salary']])
X=df.drop(columns='left')
Y=df[['left']]
df.info()
X.shape
Y.shape
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=25)
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
score=accuracy_score(Y_test,y_pred)
print(score)

*/
```

## Output:

<img width="1680" alt="Screenshot 2025-05-22 at 7 12 03 PM" src="https://github.com/user-attachments/assets/9ae2e20f-ccbc-4f74-a701-1e942a2b0072" />
<img width="1680" alt="Screenshot 2025-05-22 at 7 12 09 PM" src="https://github.com/user-attachments/assets/b1a17536-211a-4120-b60d-61c47d4b0acc" />
<img width="1680" alt="Screenshot 2025-05-22 at 7 12 14 PM" src="https://github.com/user-attachments/assets/8d0ba3f4-4e09-4964-a8e0-43e6afe1c774" />
<img width="1680" alt="Screenshot 2025-05-22 at 7 12 18 PM" src="https://github.com/user-attachments/assets/1ba5e8d5-be70-4d97-a747-626365022af5" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
