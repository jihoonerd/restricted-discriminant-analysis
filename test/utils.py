def split_X_y(df, target='Class Label'):
    X = df.loc[:, df.columns != target]
    y = df[target]
    return X, y