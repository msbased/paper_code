import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier

df_tpm = pd.read_csv('../data/NGS_tpm_info.csv')
df_target = pd.read_csv('../data/feature_importance.csv')
target_list = df_target['Unnamed: 0'].tolist()
df_tpm = df_tpm[['condition']+target_list]
X_train = df_tpm[target_list]
y_train=df_tpm['condition']

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

modelNameList = ['LogisticRegression']
np.random.seed(0)   

def classfier(modelName):
    if modelName == 'LogisticRegression':
        return LogisticRegression(C=10,max_iter=1000,class_weight='balanced')
    else:
        print('name error!')
model = []
acc_mean = []
acc_std = []
sel_feature = []

num = []
rf = pd.DataFrame()
sing_score = pd.DataFrame()
for _model in modelNameList:
    for i in range(1,113):
        clf = classfier(_model)
        rfe = RFE(estimator=clf,n_features_to_select=i,step=1)
        rfe = rfe.fit(X_train,y_train)
        sel_feature_ = X_train.columns[rfe.support_].tolist()
        sel_feature.append(sel_feature_)

        train_sel = rfe.transform(X_train)

        scores = cross_val_score(clf,train_sel,y_train,cv=3,scoring='accuracy')
        acc_mean.append(scores.mean())
        acc_std.append(scores.std())
        model.append(_model)
        num.append(i)      
       
        
    parameter = pd.DataFrame({'cross_acc_mean':acc_mean,'cross_acc_standard':acc_std, 'number':num,
                             "sel_feature":sel_feature})

max_cross_acc_mean = parameter['cross_acc_mean'].max()

selected_features = parameter.query("cross_acc_mean==@max_cross_acc_mean").iloc[0]['sel_feature']
X = df_tpm[selected_features]
y = df_tpm['condition']
model = RandomForestClassifier(random_state=0,max_depth = 2,n_estimators=100,class_weight='balanced')
model.fit(X, y)

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({'feature': model.feature_names_in_, 'importance': importances})

top10_features = feature_importance_df.nlargest(10, 'importance')

top10_features.to_csv('../result/top10_features.csv')
        