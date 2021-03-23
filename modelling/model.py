import typing as t

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "regressor": RandomForestRegressor,
        "selector": BikeRentalFeatureSelector,
        "extractor": BikeRentalFeatureExtractor   
    }

class BikeRentalFeatureSelector(BaseEstimator, TransformerMixin):
    
    
    def fit(self, X, y=None):
        self.selected_features = ['season','yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt - 24hr']
        return self
    
    def transform(self,X):
        
        return X[self.selected_features]

class BikeRentalFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self,X,y=None):
        self.mean = X['cnt'].mean()
        return self
    
    def transform(self,X):
        X['cnt - 24hr'] = [self.mean]*24 + X['cnt'].to_list()[:-24]
        
        return X


class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    _categorical_columns = "season,yr,mnth,hr,weekday,weathersit".split(",")

    _binary_columns = "holiday,workingday".split(",")

    _float_columns = "temp,atemp,hum,windspeed".split(",")

    _ignored_columns = "instant,dteday,casual,registered".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("droper", "drop", type(self)._ignored_columns),
                ("binarizer", OrdinalEncoder(), type(self)._binary_columns),
                (
                    "one_hot_encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    type(self)._categorical_columns,
                ),
                ("scaler", StandardScaler(), type(self)._float_columns),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)







class AgeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAddAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        return X



class SimplifiedTransformer(BaseEstimator, TransformerMixin):
    """This is just for easy of demonstration"""

    _columns_to_keep = "HouseAge,GarageAge,LotArea,Neighborhood,HouseStyle".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("binarizer", OrdinalEncoder(), ["Neighborhood", "HouseStyle"]),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        self._column_transformer = self._column_transformer.fit(X_, y=y)
        return self

    def transform(self, X):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        X_ = self._column_transformer.transform(X_)
        return X_
