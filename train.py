#%%
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from feature_engine import discretisation, encoding

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import pipeline, metrics, ensemble, tree


# %%
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=728709897679564592)


# %%
raw_data = pd.read_csv('./data/abt_churn.csv')
raw_data.head()


# %%
oot = raw_data[raw_data['dtRef'] == raw_data['dtRef'].max()].copy()
df = raw_data[raw_data['dtRef'] < raw_data['dtRef'].max()].copy()


# %%
features = df.iloc[:,2:-1].columns
target = 'flagChurn'


# %%
X = df[features]
y = df[target]


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)


# %%
"""Checkign missing values on the trainning dataset"""
X_train.isna().sum().sort_values(ascending=False)


# %%
analysis_data = X_train.copy()
analysis_data[target] = y_train


# %%
analysis_data.info()

# %%
summary = analysis_data.groupby(by=target).agg(['mean', 'median']).T
summary


# %%
summary['diff'] = summary[0] - summary[1]
summary['diff_rel'] = summary[0] / summary[1]
summary.sort_values(by=['diff_rel'], ascending=False)


"""Decision Tree to identify the best features in the dataset"""
tree_features = tree.DecisionTreeClassifier(
    random_state=42
)

tree_features.fit(X_train, y_train)

# %%
features_importance = pd.Series(tree_features.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()
features_importance['acum.'] = features_importance[0].cumsum()

# %%
best_features = features_importance[features_importance['acum.'] <= 0.97]['index'].to_list()
best_features


# %%
"""
    Since we working with continuos variables we
    can use the discretisation for replace the number itself 
    for the respective bin.
"""

tree_discretisation = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    cv=3,
    regression=False,
    random_state=42,
    bin_output='bin_number'
)

tree_discretisation.fit(X_train, y_train)


onehot = encoding.OneHotEncoder(
    variables=best_features,
    ignore_format=True
)



# %%
with mlflow.start_run():

    mlflow.sklearn.autolog()

    model = ensemble.RandomForestClassifier(
        random_state=42,
        n_jobs=2
    )


    params = {
        'criterion':['gini', 'entropy', 'log_loss'],
        'min_samples_leaf':[15, 20, 25, 30, 50, 60],
        'n_estimators':[100, 200, 500, 1000]
    }

    grid = GridSearchCV(
        model,
        param_grid=params,
        cv=3,
        scoring='roc_auc',
        verbose=4
    )


    model_pipeline = pipeline.Pipeline(
        steps=[
            ('Discretisation',tree_discretisation),
            ('Onehot',onehot),
            ('Grid',grid)
        ]
    )

    model_pipeline.fit(X_train[best_features], y_train)


    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_predict_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_predict_proba)
    roc_train = metrics.roc_curve(y_train, y_train_predict_proba)

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_predict_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_predict_proba)
    roc_test = metrics.roc_curve(y_test, y_test_predict_proba)

    oot_predict = model_pipeline.predict(oot[best_features])
    oot_predict_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], oot_predict_proba)
    roc_ott = metrics.roc_curve(oot[target], oot_predict_proba)


    mlflow.log_metrics(
        {
            'acc_train':acc_train,
            'auc_train':auc_train,
            'acc_test':acc_test,
            'auc_test':auc_test,
            'acc_oot':acc_oot,
            'auc_oot':auc_oot
        }
    )


# %%
