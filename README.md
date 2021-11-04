# restricted-discriminant-analysis
RDA implementation compatible with Scikit-learn API

## How to use

Model class is inherited from `BaseEstimator` of `scikit-learn`, so you can use the `RDA` model just like the other `scikit-learn`'s esimators:

`RDA` uses macro-f1 score as a score function.

### Model Only

```python
rda_model = RDA()
rda_model.fit(X_train, y_train)
preds = rda_model.predict(X_test)
```

### With Pipeline

```python
# Gridsearch CV
parameters = {'rda__alpha': np.linspace(0, 1.0, 11), 'rda__beta':np.linspace(0, 1.0, 11), 'rda__variance': [0.1, 0.5, 1.0]}
pipeline = Pipeline(
    steps = [('scaler', StandardScaler()), ('rda', RDA())]
)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
rda_g_search = GridSearchCV(pipeline, parameters, cv=skf, n_jobs=-1, verbose=1)
rda_g_search.fit(X.to_numpy(), y.to_numpy())

```
