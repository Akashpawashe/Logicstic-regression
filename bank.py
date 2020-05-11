

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report

bank= pd.read_csv("D:\\excelR\\Data science notes\\Logistic Regression\\agmnt\\bank-full.csv",sep=";")

#In dataset some variables has no importance and unkown data so droppin it
bank.drop(["education"],inplace=True,axis=1)
bank.drop(["pdays"],inplace=True,axis=1)
bank.drop(["previous"],inplace=True,axis=1)
bank.drop(["poutcome"],inplace=True,axis=1)
bank.drop(["month"],inplace=True,axis=1)
bank.drop(["contact"],inplace=True,axis=1)
bank.drop(["job"],inplace=True,axis=1)
bank.shape

#converting into binary
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
bank["y"]=LE.fit_transform(bank["y"])
bank.columns
bank["marital"]=LE.fit_transform(bank["marital"])
bank["default"]=LE.fit_transform(bank["default"])
bank["housing"]=LE.fit_transform(bank["housing"])
bank["loan"]=LE.fit_transform(bank["loan"])

#EDA
a1=bank.describe()
bank.median()
bank.var()
bank.skew()
plt.hist(bank["age"])
plt.hist(bank["balance"])
plt.hist(bank["duration"])
plt.show
bank.isna().sum()#No null values present
bank.isnull().sum()

bank.y.value_counts()
bank.loan.value_counts()
bank.housing.value_counts()
cor=bank.corr()
cor

sb.boxplot(x="y",y="age",data=bank,palette="hls")
sb.boxplot(x="y",y="balance",data=bank,palette="hls")
bank.shape
 
### Splitting the data into train and test data 
train,test  = train_test_split(bank,test_size = 0.3) 
train.columns



#model buliding
model1=sm.logit('y~age+marital+default+balance+housing+loan+day+duration+campaign',data=train).fit()
model1.summary()#Housing variables are insignificant, so remove and build new model
model1.summary2() ## r sqr=0.216
   
#####feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model1 = ExtraTreesClassifier()
model1 = ExtraTreesClassifier()

X = bank.iloc[:,:9]  #independent columns
y = bank.iloc[:,-1]    #target column i.e Y
model1.fit(X,y)

print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model1.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

##from  feature selection we canm drop following columns
bank.drop(["marital","default","housing","loan"],inplace=True,axis=1)
X = bank.iloc[:,:5]  #independent columns
y = bank.iloc[:,-1]    #target column i.e Y
model1.fit(X,y)

#new model after droping columns from feature importance
model2=sm.logit('y~age+balance+day+duration+campaign',data=train).fit()
model2.summary2()# r sqr= 0.171


import seaborn as sns
#get correlations of each features in dataset
corrmat = bank.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(bank[top_corr_features].corr(),annot=True,cmap="RdYlGn")

### Splitting the data into train and test data 
train,test  = train_test_split(bank,test_size = 0.2) 
train.columns


#prediction
train_pred = model1.predict(train.iloc[:,:-1])
# Creating new column 

# filling all the cells with zeroes
train.reset_index(inplace =True)
train["train_pred"] = np.zeros(36168)

train.loc[train_pred>0.5,"train_pred"] = 1

classification = classification_report(train["train_pred"],train["y"])


#confusion matrix
confusion_matrx = pd.crosstab(train.train_pred,train['y'])
confusion_matrx

accuracy_train = (31950+4218)/(31950+4218)
print(accuracy_train)#88.97

#ROC CURVE AND AUC
fpr,tpr,threshold = metrics.roc_curve(train["y"], train_pred)

#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")




