
# %%
import pandas as pd
import mlflow
import mlflow.sklearn


# %%
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=728709897679564592)


# %%
models = mlflow.search_registered_models(filter_string='name = "model_churn"')

# %%
latest_version = max([i for i in models[0].latest_versions[0].version])

# %%
model = mlflow.sklearn.load_model(f'models:/model_churn/{latest_version}')
features = model.feature_names_in_

# %%
df = pd.read_csv('./data/abt_churn.csv')
sample = df[df['dtRef'] == df['dtRef'].max()].sample(10)
sample

# %%
predict = model.predict_proba(sample[features])[:,1]
sample['predict'] = predict
sample[['flagChurn', 'predict']]
