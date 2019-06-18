import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def read_json():
    # read time series
    ts = pd.read_csv(
        'daily-min-temperature.csv',
        parse_dates=["Date"],
        index_col=0,
        squeeze=True
    )
    return ts


class TimeSeriesRegressor:
    def __init__(self, ts, granularity=0, num_lags=60, train_test_size=0.33):
        self.num_lags = num_lags
        self.granularity = granularity
        self.is_trained = False
        self.ts = ts
        self.train_test_size = train_test_size

    def _make_lags_matrix(self):
        lags_matrix = pd.DataFrame(index=self.ts[self.num_lags:].index)
        target = self.ts[self.num_lags:]
        lags_matrix['target'] = target
        for i in range(1, self.num_lags + 1):
            lag_name = f'lag_{i}'
            lags_matrix[lag_name] = self.ts.shift(i)[self.num_lags:]
        return lags_matrix

    def _feature_enrichment(self, lags_matrix):
        # static features
        lags_matrix['std'] = lags_matrix.drop(['target'], axis=1).apply(
            lambda x: x.std(), axis=1)
        lags_matrix['mean'] = lags_matrix.drop(['target'], axis=1).apply(
            lambda x: x.mean(), axis=1)
        # datetime features
        lags_matrix['year'] = lags_matrix.index.map(lambda x: x.year)
        lags_matrix['month'] = lags_matrix.index.map(lambda x: x.month)
        lags_matrix['day'] = lags_matrix.index.map(lambda x: x.day)
        return lags_matrix

    def train(self, train_type: int = 0):
        if train_type == 1:
            self._train_gradient_boosting()
        else:
            self._train_linear_regression()

    def _train_linear_regression(self):
        prepared_param = self._prepare()
        self.t_model = LinearRegression(n_jobs=-1)
        self.t_model.fit(prepared_param['X_train'], prepared_param['y_train'])
        y_pred = self.t_model.predict(prepared_param['X_test'])
        print(r2_score(prepared_param['y_test'], y_pred))
        prepared_param['y_test'].plot(figsize=(20, 20))
        pd.Series(y_pred, index=prepared_param['y_test'].index).plot()
        feature_importances = {
            feature: feature_value
            for feature, feature_value
            in zip(prepared_param['X'].columns, self.t_model.coef_)
        }
        print(sorted(feature_importances.items(), key=lambda x: x[1]))

    def _train_gradient_boosting(self):
        prepared_param = self._prepare()
        self.t_model = GradientBoostingRegressor()
        self.t_model.fit(prepared_param['X_train'], prepared_param['y_train'])
        self.is_trained = True
        y_pred = self.t_model.predict(prepared_param['X_test'])
        print(r2_score(prepared_param['y_test'], y_pred))
        prepared_param['y_test'].plot(figsize=(20, 20))
        self.plotting = pd.Series(y_pred, index=prepared_param['y_test'].index)
        feature_importances = {
            feature: feature_value
            for feature, feature_value
            in zip(prepared_param['X'].columns, self.t_model.feature_importances_)
        }
        print(sorted(feature_importances.items(), key=lambda x: x[1]))

    def predict(self, n=1):
        pass

    def _prepare(self):
        prepared = {}
        prepared['feature_matrix'] = self._feature_enrichment(
            self._make_lags_matrix())
        prepared['X'], prepared['y'] = prepared['feature_matrix'].drop(
            ['target'], axis=1), prepared['feature_matrix']['target']
        train_dictionary = ['X_train', 'X_test', 'y_train', 'y_test']
        train_tuple = train_test_split(
            prepared['X'],
            prepared['y'],
            test_size=self.train_test_size
        )
        prepared.update(zip(train_dictionary, train_tuple))
        return prepared


trained_model = TimeSeriesRegressor(read_json())
trained_model.train(1)
# finish
print('The end')
