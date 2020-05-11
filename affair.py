



#logistic regression
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report

#loading the data
affair=pd.read_csv("D:\\excelR\\Data science notes\\Logistic Regression\\agmnt\\affairs.csv")

# Droping first column 
affair.drop(["sr"],inplace=True,axis = 1)    
affair=pd.get_dummies(affair)

#converting affair variable into binary
affair.loc[affair.affairs>0,'affairs']=1

#EDA
a1=affair.describe()
affair.median()
affair.var()
affair.skew()
plt.hist(affair["age"])
plt.hist(affair["yearsmarried"])


affair.isna().sum()#No NA values present, so no need to do imputation


affair.affairs.value_counts()#count 0 and 1s
affair.gender_female.value_counts()
affair.gender_male.value_counts()
affair.children_no.value_counts()
cor=affair.corr()

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sb.boxplot(x="affairs",y="age",data=affair,palette="hls")
sb.boxplot(x="affairs",y="yearsmarried",data=affair,palette="hls")

### Splitting the data into train and test data 
aff_train,aff_test  = train_test_split(affair,test_size = 0.3) # 30% size
aff_train.columns


#model buliding
 
model1=sm.logit('affairs~age+yearsmarried+religiousness+education+occupation+rating+gender_female+gender_male+children_no+children_yes',data=aff_train).fit()
model1.summary()#many variables are insignificant, so remove and build new model
model1.summary2()

#####feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model1 = ExtraTreesClassifier()
model1 = ExtraTreesClassifier()

X = affair.iloc[:,1:13]  #independent columns
y = affair.iloc[:,0]    #target column i.e Y
model1.fit(X,y)

print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model1.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
affair.columns
##from  feature selection we canm drop following columns
affair.drop(["dummy_v1","gender_female","gender_male","dummy_v2"],inplace=True,axis=1)
X = affair.iloc[:,1:9]  #independent columns
y = affair.iloc[:,-1]    #target column i.e Y
model1.fit(X,y)

#new model after droping columns from feature importance
model2=sm.logit('affairs~age+yearsmarried+religiousness+education+occupation+rating+children_no+children_yes',data=aff_train).fit()
model2.summary2()# r sqr= 0.171


import seaborn as sns
#get correlations of each features in dataset
corrmat = affair.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(affair[top_corr_features].corr(),annot=True,cmap="RdYlGn")


#from heatmap we arwe removing below columns
aff_train.drop(["education"],inplace=True,axis=1)
aff_train.drop(["occupation"],inplace=True,axis=1)

#build new model
model2= sm.logit('affairs~age+yearsmarried+religiousness+rating+children_no+children_yes',data=aff_train).fit()
model2.summary()
model2.summary2()

####model2 is better than model1 

#prediction
train_pred = model2.predict(aff_train.iloc[:,1:])
# Creating new column for storing predicted y

# filling all the cells with zeroes
aff_train["train_pred"] = np.zeros(420)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
aff_train.loc[train_pred>0.5,"train_pred"] = 1

#classification report
classification = classification_report(aff_train["train_pred"],aff_train["affairs"])
'''
              precision    recall  f1-score   support

         0.0       0.96      0.77      0.86       398
         1.0       0.10      0.45      0.16        22

    accuracy                           0.75       420
   macro avg       0.53      0.61      0.51       420
weighted avg       0.92      0.75      0.82       420
'''
#confusion matrix
confusion_matrx = pd.crosstab(aff_train.train_pred,aff_train['affairs'])
confusion_matrx

accuracy_train = (299+23)/(299+23+82+16)
print(accuracy_train)#76.57

#ROC CURVE AND AUC
fpr,tpr,threshold = metrics.roc_curve(aff_train["affairs"], train_pred)

#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

################ AUC #########################

roc_auc = metrics.auc(fpr, tpr)         #0.705 : Good model

######################It is a good model with AUC = 0.705 ###############################


#Based on ROC curv we can say that cut-off value = 0.50 is the best value for higher accuracy , by selecting different cut-off values accuracy is decreasing.

# Prediction on Test data set

test_pred = model2.predict(aff_test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
aff_test["test_pred"] = np.zeros(181)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
aff_test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(aff_test.test_pred,aff_test['affairs'])

confusion_matrix
accuracy_test = (129+8)/(129+8+37+7) 
accuracy_test#75.58


'''
####### Its a Just right model because Test and Train accuracy is nearly same #################

Train accuracy=76.47
Test accuracy=75.58

'''