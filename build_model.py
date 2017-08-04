import xgboost as xgb
import math
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder();

df = pd.read_csv('./train_shuf.csv', header=0);
df = df.drop(['id'], axis=1);
df['target'] = le.fit_transform(df['target']);
ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=[1 if col not in ['target'] else 0 for col in df.columns]);
ohe.fit(df);

train, val = train_test_split(df, test_size=0.3);

train_y = train.target;
train_x = train.drop(['target'], axis=1);
train_x = ohe.transform(train_x);

val_y = val.target;
val_x = val.drop(['target'], axis=1);
val_x = ohe.transform(val_x);

df_test = pd.read_csv('./test.csv', header=0);
df_test = df_test.drop(['id'], axis=1);
df_test = ohe.transform(df_test);

params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'reg_lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
'seed':1000,
'max_depth':13,
'learning_rate': 0.4,
'nthread':7,# cpu 线程数
'n_estimators':50
};


print('train_x.shape=' + str(train_x.shape));
print('val_x.shape=' + str(val_x.shape));
print('df_test.shape=' + str(df_test.shape));
clf = xgb.XGBClassifier(learning_rate=0.05, n_estimators=100, min_child_weight=30, max_depth=15, reg_lambda=1, objective='binary:logistic');
clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], eval_metric='mlogloss', early_stopping_rounds=5);

preds = clf.predict_proba(df_test);

clf._Booster.dump_model('./xgb.model');

fo = open('./submit.txt', 'w');
fo.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n');
for i,v in enumerate(preds):
    fo.write(str(i+1) + ',' + ','.join([str(f) for f in v]) + '\n');
fo.close();
