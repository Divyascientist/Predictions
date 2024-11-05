import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,RepeatedStratifiedKFold
from skopt import BayesSearchCV
import logging 

logging.basicConfig(
    filename='KNN.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
                   )

df=pd.read_csv("titanic_test.csv")
df.drop(['Name','Cabin','Ticket','PassengerId'],axis=1,inplace=True)


#seperate out numeric and categorical values

df_num=df.select_dtypes(exclude='object')
df_obj=df.select_dtypes('object')

## missing value treatment
df_num=df_num.apply(lambda x: x.fillna(x.mean()),axis=1)
df_obj=df_obj.apply(lambda x: x.fillna(x.mode()[0]),axis=1)

##outlier treatment
df_num=df_num.apply(lambda x:x.clip(upper=x.quantile(0.995),lower=x.quantile(0.05)))

##Encoding categorical variables
df_obj=pd.get_dummies(df_obj,dtype=int,drop_first=True)

df_processed=pd.concat([df_num,df_obj],axis=1)

df_processed.to_csv("df_processed.csv")
df_num.to_csv("df_num.csv")

##strain test plit
train_x,test_x,train_y,test_y=train_test_split(df_processed.drop(['Pclass'],axis=1),df_processed['Pclass'],test_size=0.2,random_state=12)

## KNN
def knn_model(k=10):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x,train_y.astype('int'))

    ##predict

    pred_train_y=knn.predict(train_x)
    pred_test_y=knn.predict(test_x) 

    ## error calculation 
    return(accuracy_score(train_y,pred_train_y),accuracy_score(test_y,pred_test_y))  

for i in range(5,20):
    tr_acc,te_acc=knn_model(i)
    print("#"*20)
    print(i)
    print ("accuracy score in training data is:",tr_acc )
    print ("accuracy score in test data is:" ,te_acc)

##grid 

knn=KNeighborsClassifier()
grid_knn={'n_neighbors': list(range(21,30))}
knn_clf= GridSearchCV(knn, grid_knn, cv=10, scoring='accuracy')
knn_clf.fit(train_x,train_y.astype('int'))

logging.info(" !!!!!!!!!  Grid search ")
logging.info(knn_clf.best_params_)

knn=KNeighborsClassifier(n_neighbors=knn_clf.best_params_['n_neighbors'])
knn.fit(train_x,train_y.astype('int'))

    ##predict

pred_train_y=knn.predict(train_x)
pred_test_y=knn.predict(test_x) 

("Training ==" ,accuracy_score(train_y.astype('int'),pred_train_y),"//// Testing==",accuracy_score(test_y.astype('int'),pred_test_y))


##Randomised search
knn=KNeighborsClassifier()
grid_knn={'n_neighbors': list(range(2,300))}
knn_clf=RandomizedSearchCV(knn,grid_knn,cv=10,scoring='accuracy',n_iter=10,n_jobs=-1,random_state=123)
knn_clf.fit(train_x,train_y.astype('int'))

logging.info(" !!!!!!!!!  Random search ")
logging.info(knn_clf.best_params_)


##baegian  search

params=dict()
params['n_neighbors']=(2,100,'uniform')
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
search=BayesSearchCV(estimator=KNeighborsClassifier(),search_spaces=params,n_jobs=-1,cv=cv)
search.fit(train_x,train_y)
logging.info(" !!!!!!!!!  baiegin search ")
logging.info(search.best_params_)


