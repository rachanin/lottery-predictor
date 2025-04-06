import pandas as pd

def create_features(df, window_size=10):
    numbers = df['numbers'].values
    features = []
    targets = []

    for i in range(window_size, len(numbers)):
        features.append(numbers[i-window_size:i])
        targets.append(numbers[i])

    X = pd.DataFrame(features)
    y = pd.Series(targets)
    return X, y
