import numpy as np
import pandas as pd
from rda.rda_model import RDA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from test.utils import split_X_y


def test_rda_model():
    df = pd.read_csv('sample_data/facial_expression_train_dataset.csv')
    target = 'Class Label'
    features = df.loc[:, df.columns != target]

    print("[Data Shape]")
    print(f"Full Dataset: {df.shape}")
    print(f"Features: {features.shape}")

    X, y = split_X_y(df, target=target)

    
    parameters = {'alpha': np.linspace(0, 1.0, 11), 'beta':np.linspace(0, 1.0, 11), 'variance': [0, 0.1, 1.0, 2.0]}

    # Gridsearch CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    rda = RDA()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    rda_g_search = GridSearchCV(rda, parameters, cv=skf, n_jobs=-1)
    rda_g_search.fit(X_scaled, y.to_numpy())