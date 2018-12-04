# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:10:21 2018

@author: hp
"""
import pandas as pd
import numpy as np
#%%
#header=0 if it has header
#delimiter=' *, *' to remove the leading and trailing spaces between the values for csv,tab seperated files delimiter='\t'
df=pd.read_csv(r'C:\Users\hp\Desktop\DSP\adult_data.csv',
               header=None,delimiter=' *, *',engine='python')
#%%
df.head()
#%%
#to display all the columns in a large dataset
pd.set_option('display.max_columns',None)
df.head()
#%%
#dimensions of dataset
df.shape
#%%
#assigning column names to the columns
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

df.head()
#%%
#to check for missing vlaues
df.isnull().sum()
df=df.replace(['?'],np.nan)
df.isnull().sum()
#%%
#make copy of the dataset so that the modifications are not done on the original dataset
df_rev=pd.DataFrame.copy(df)
df_rev.describe(include='all')
#%%
#for loop to replace all missing values 
#mode[0] to access the index value
for value in ['workclass','occupation','native_country']:
    df_rev[value].fillna(df_rev[value].mode()[0],inplace=True)
df_rev.isnull().sum()
#%%
#to fill all the missing values in categorical data in entire data set with its mode
#for x in df_rev.columns[:]:
 #   if df_rev[x].dtype=='object':
  #      df_rev[x].fillna(df_rev[x].mode()[0],inplace=True)
#%%
col=['workclass','education','marital_status','occupation',
     'relationship','race','sex','native_country','income']
col
#%%
#converting categorial to numerical
from sklearn import preprocessing
le={}

for x in col:
    le[x]=preprocessing.LabelEncoder()
    
for x in col:
    df_rev[x]=le[x].fit_transform(df_rev[x])
#%%
df_rev.head()
#<=50k===> 0
#>=50k===> 1
#%%
df_rev.dtypes
#%%
#creating x and y variables
X=df_rev.values[:,:-1]
Y=df_rev.values[:,-1]
#%%
#Scaling data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X)

X=scaler.transform(X)
print(X)
#display entire array
np.set_printoptions(threshold=np.inf)
#to avoid a certain error
Y=Y.astype(int)
#%%
##creating train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#%%
#model creation
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print('classification report:')

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc)
#####################recall
#For class0 -(how many -ive cases did we catch)
#(specificity or true negative rate)=TN/(TN+FP)=0.94

#For class1-(how many positive cases did we catch)
#(Sensitivity or true positive rate)=TP/TP+FN=0.44

#####################precision
#For class0-(how many of the negative predictions were correct)
#=TN/(TN+FN)=0.84
#for class1-(how many of the positive predictions were correct)
#=TP/(TP+FP)=0.71

#######################f1 score
#harmonic mean of precision and recall
#formula=2*(precision*recall)/(Precision + recall)

#support(number of 0s and 1s)
#%%
#lets increase the accuracy


#store the predicted probabilities
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)
#%%
#setting threshold for list
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(y_pred_class)
#%%
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)

print('classification report:')

print(classification_report(Y_test,y_pred_class))
acc=accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model:",acc)
#%%
#type1 and type 2 errors for each threshold in for loop
for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0) #where is like ifelse loop in R
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0] #type1 +type2 error
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", \
    cfm[1,0]," , type 1 error:", cfm[0,1])
#cfm[1,0]=type1 error
#cf[0,1]=type2 error
#total error should be less along with the type 2 error
    
#if client requirement is less type 2 error then we can
 #go with 0.3 threshold in our case we don have such conditions
  #so we can go with the leaset totle error that is 0.45

#%%
#roc for specific threshold
#ROC(Receiver operating characteristics) curve CURVE $ AUC(Area under the curve) VALUE
#fpr (false positive rate) tpr(true positive rate)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(Y_test,y_pred_class)
auc=metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
#%%
#plotting roc curve

import matplotlib.pyplot as plt
plt.title('Receiver Operating Charateristics')
#b indicates blue line
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='lower right')
#r-- means red line with dash lines
plt.plot([0,1],[0,1],'r--')
#limit x and y axis scale to 0 to 1
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#auc=0.5-0.6=poor
#0.6-0.7=bad
#0.7-0.8=good
#0.8-0.9=v.good
#0.9-1=excellent
#%%
#########for overall model with specifying threshold

from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(Y_test,y_pred_prob[:,1])
auc=metrics.auc(fpr,tpr)
print(auc)

#%%
#plotting roc curve

import matplotlib.pyplot as plt
plt.title('Receiver Operating Charateristics')
#b indicates blue line
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='lower right')
#r-- means red line with dash lines
plt.plot([0,1],[0,1],'r--')
#limit x and y axis scale to 0 to 1
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%%
#K-fold validaton
#Using cross validation

classifier=(LogisticRegression())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#for going ahead with the cross validation model
for train_value, test_value in kfold_cv: 
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
#%%
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print('classification report:')

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc)
#%%
#FEATURE SELECTION
#Most preferred|
#1.RECURSIVE feature elimiation(RFE)(similar to backward selection)
#arguments---RFE(classifier,(number of features to be retained(for eg 7)),)



#2.Univariate feature selection(SelectBest,SelectPercentile)
#uses chi-square,selects variables with the best chisquare values

from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
colname=df_rev.columns[:]


from sklearn.feature_selection import RFE
rfe=RFE(classifier,10)
model_rfe=rfe.fit(X_train,Y_train)
print("num feature",model_rfe.n_features_)
print("selected Features")
print(list(zip(colname,model_rfe.support_)))
print("feature ranking",model_rfe.ranking_)

Y_pred=model_rfe.predict(X_test)
#%%
#confusion matix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print('classification report:')

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc)
#%%
#Univariate selection
X=df_rev.values[:,:-1]
Y=df_rev.values[:,-1]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=11 )
fit1 = test.fit(X, Y)

print(fit1.scores_) #returns set of chi2 values
print(list(zip(colname,fit1.get_support()))) #returns boolean list of values(true/false)
features = fit1.transform(X)

 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(features)

X = scaler.transform(features)
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#%%
#model creation
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print('classification report:')

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc)