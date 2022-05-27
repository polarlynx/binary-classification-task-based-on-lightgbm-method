import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns
import gc
num_round = 1500

# load processed training data and test data
train_data = pd.read_csv('final_train.csv').drop(['label','confidence'], axis=1)
train_label = pd.read_csv('final_train.csv',usecols=['label'])
train=pd.concat([train_data,train_label],axis=1)
features = [i for i in train_data.columns]
test_data = pd.read_csv('final_test.csv')

train_x = train_data
train_y = train_label.values
test =np.array(test_data)

# set the parameters
params = {'num_leaves': 40,
          'min_data_in_leaf': 30,
          'objective': 'binary',
          'max_depth': 5,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 1,
          "bagging_freq": 1,
          'min_child_samples': 18,
          'min_child_weight': 0.001,
          "bagging_fraction": 1,
          "bagging_seed": 11,
          "lambda_l1": 0.001,
          'lambda_l2': 4,
          "verbosity": -1,
          "nthread": -1,
          'metric': {'binary_logloss', 'auc'},
          "random_state": 2019,
          # 'device': 'gpu'
          }

# k-fold cross validation
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], ))
test_pred_prob = np.zeros((test.shape[0], ))


## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])

    evals_result = {}  # record training results
    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    evals_result=evals_result,
                    verbose_eval=20,
                    #categorical_feature=features,
                    early_stopping_rounds=60)
    print('plot AUC curves...')
    ax = lgb.plot_metric(evals_result, metric='auc')  # metric的值与之前的params里面的值对应
    plt.savefig('pic/auc_fold {}.png'.format(fold_ + 1), dpi=600)
    plt.show()
    print('save pic fold_auc {} done!'.format(fold_ + 1))
    print('plot loss curves...')
    ax = lgb.plot_metric(evals_result, metric='binary_logloss')  # metric的值与之前的params里面的值对应
    plt.savefig('pic/loss_fold {}.png'.format(fold_ + 1), dpi=600)
    plt.show()
    print('save pic_loss fold {} done!'.format(fold_ + 1))

    # calculate feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    # predict test data
    test_pred_prob += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

# Generate prediction file
threshold = 0.5
res=[]
for pred in test_pred_prob:
    result = 1 if pred > threshold else 0
    res.append(result)

data1 = pd.DataFrame(res,columns=['prediction'])
data1.to_csv('prediction.csv',index=None)
print('done!')

## plot feature importance
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:10].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('pic/feature_importances.png')