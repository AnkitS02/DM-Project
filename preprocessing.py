import pandas as pd
import matplotlib.pyplot as mt
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True) #prevent numpy exponential
pd.set_option('display.max_columns', 16)

global ncols
ap_data=pd.read_excel("dataset.xlsx")

ncols=ap_data.columns
print(ncols)

des=ap_data.describe()
print(des)

raw_data=pd.DataFrame(ap_data)
raw_data=raw_data.dropna() #handling missing values
data_with_out=raw_data.drop_duplicates() #handling duplicate values

print("data with outliers:")
print(data_with_out)

summery=data_with_out.describe()
print(summery)

data_without_out = pd.DataFrame()

def outliers_iqr(df,column): #removing outliers using IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqrange=q3-q1
    min=q1-iqrange*1.5
    max=q3+iqrange*1.5
    mask = df[column].between(min, max, inclusive=True)
    iqr = df.loc[mask, column]
    return iqr

for i in range(3,len(ncols)):
    #printing graphs with outliers
    x = data_with_out.loc[:,ncols[i]]
    sns.boxplot(x,None)
    mt.title("with outliers " + ncols[i])
    mt.xlabel(ncols[i])
    mt.ylabel('frequency')
    mt.savefig('graphs/with_outliers/'+str(i))
    mt.close()

    temp = outliers_iqr(data_with_out,ncols[i])
    data_without_out=data_without_out.append(temp)

data_without_out = data_without_out.transpose()
data_without_out = data_without_out.dropna()  # handling missing values if any after outliers are removed
data_without_out = data_without_out.drop_duplicates()  # handling duplicate values if any after outliers are removed
print("data without outliers:")
print(data_without_out)

for i in range(3, len(ncols)):
    #printing graphs without outliers
    x = data_without_out.loc[:,ncols[i]]
    fig=sns.boxplot(x,None)
    mt.title("without outliers " + ncols[i])
    mt.xlabel(ncols[i])
    mt.ylabel('frequency')
    mt.savefig('graphs/without_outliers/'+str(i))
    mt.close()

array=data_without_out.values
print(array)
print("data skewness:")
print(data_without_out.skew())

X = array[:,0:9]
Y= array[:,9]
train_X,test_X,train_y,test_y=tts(X,Y,random_state=6,test_size=0.2) #data split, test size 20%
model= LinearRegression()
result= model.fit(train_X,train_y)
result.fit(train_X,train_y)
pred=result.predict(test_X)
print("model's coefficient")
print(model.coef_)
print("model's intercept")
print(model.intercept_)
z=r2_score(test_y,pred)
print("r2_score: ",z)
print("accuracy: ",model.score(X,Y)*100,"%")

path="C:\\Users\\06atu\\Desktop\\dm_project\\scatterplot\\"+str(ncols[12])+"\\"
os.makedirs(path)
for j in range(3,len(ncols)-1):
    ay = sns.regplot(x=ncols[j], y=ncols[12], data=data_without_out)
    mt.title("Scatter plot of " + ncols[12]+'with other attributes')
    mt.xlabel(ncols[12])
    mt.ylabel(ncols[j])
    mt.savefig(path+str(j))
    mt.clf()
    mt.cla()
print("cross validation using test set as sample to train model")
model= LinearRegression()
result= model.fit(test_X,test_y)
result.fit(test_X,test_y)
pred=result.predict(test_X)
print("model's coefficient")
print(model.coef_)
print("model's intercept")
print(model.intercept_)
z=r2_score(test_y,pred)
print("r2_score: ",z)
print("accuracy: ",model.score(X,Y)*100,"%")

print("using k folds:")
scores=cross_val_score(model,train_X,train_y,cv=10)
print("scores: ",scores)
print("mean: ",scores.mean())
print("std deviation: ",scores.std())