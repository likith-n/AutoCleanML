import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

class MissingValueHandler:
    def __init__(self, strategy="mean", n_neighbors=5):
        """
        strategy: mean | median | most_frequent | knn
        """
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer = None

    def fit(self, df: pd.DataFrame):
        if self.strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        else:
            self.imputer = SimpleImputer(strategy=self.strategy)

        self.imputer.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        data = self.imputer.transform(df)
        return pd.DataFrame(data, columns=df.columns, index=df.index)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
