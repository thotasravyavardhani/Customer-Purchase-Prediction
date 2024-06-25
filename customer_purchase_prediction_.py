#step 1: import Library
import pandas as pd

#step2: import data
purchase=pd.read_csv('Customer Purchase.csv')

purchase

purchase.head()#1st 5 rows

purchase.tail()#last 5 rows

purchase.head(3)#1st 3 rows

purchase.tail(3)#last 3 rows

purchase.info()#info about data in dataset

purchase.describe()#stats about data

purchase.columns#All colns in dataset

purchase.shape#rows and colns

#Step 3: Splitting the data into X and Y
x=purchase[['Age','Gender','Education','Review']]
y=purchase['Purchased']
#encoding data into string to categorical data(0 or 1)
x.replace({'Review':{'Poor':0,'Average':1,'Good':2}},inplace=True)
x.replace({'Gender':{'Male':0,'Female':1}},inplace=True)
x.replace({'Education':{'School':0,'UG':1,'PG':2}},inplace=True)

x

#Splitting the data into train data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

#fit or train the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train) #To train the model

#To test the data
y_pred=model.predict(x_test)

#To know accuracy
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_pred)

#Accuracy for correct = 30%
#So Anoher model-Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy_score(y_test,y_pred)

#Accuracy= 61%
#So,Lets see another method KNN-K Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier
#if y is regression, use from sklearn.neighbors import kNeighborsRegressor
model=KNeighborsClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy_score(y_test,y_pred)

#Accuracy=53%
#Another model

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy_score(y_test,y_pred)

#Accuracy= 61%

