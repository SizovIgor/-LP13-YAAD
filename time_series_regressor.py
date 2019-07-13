from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from anomaly_detector import AnomalyDetector


def read_json(file='daily-min-temperature.csv', sep=','):
    # read time series
    return pd.read_csv(
        file,
        parse_dates=['Date'],
        index_col=0,
        squeeze=True,
        sep=sep,
    )


class TimeSeriesRegressor:

    def __init__(self, granularity=24, num_lags=60, train_ratio=0.33):
        self.score = None
        self.granularity = timedelta(hours=granularity)
        self.num_lags = num_lags
        self.train_ratio = train_ratio
        self.is_trained = False
        self.t_model = None

    def _make_lags_matrix(self, ts):
        values = np.array([
            ts[row:row + self.num_lags][::-1]
            for row in range(len(ts) - self.num_lags)
        ])
        columns = ['lag_%s' % (i + 1) for i in range(self.num_lags)]
        frame = pd.DataFrame(
            columns=columns,
            data=values,
            index=ts[self.num_lags:].index
        )
        frame['target'] = ts[self.num_lags:]

        # lags_matrix = pd.DataFrame(index=ts[self.num_lags:].index)
        # target = ts[self.num_lags:]
        # lags_matrix['target'] = target
        # for num_lag in range(1, self.num_lags + 1):
        #     lag_name = f'lag_{num_lag}'
        #     lags_matrix[lag_name] = ts.shift(num_lag)[self.num_lags:]
        # return lags_matrix
        return frame

    @staticmethod
    def _feature_enrichment(lags_matrix):
        # static features
        lags_matrix['std'] = lags_matrix.drop(
            labels='target',
            axis=1,
            errors='ignore',
        ).apply(
            lambda x: x.std(),
            axis=1,
        )
        lags_matrix['mean'] = lags_matrix.drop(
            labels='target',
            axis=1,
            errors='ignore',
        ).apply(
            lambda x: x.mean(),
            axis=1,
        )
        # datetime features
        lags_matrix['year'] = lags_matrix.index.map(lambda x: x.year)
        lags_matrix['month'] = lags_matrix.index.map(lambda x: x.month)
        lags_matrix['day'] = lags_matrix.index.map(lambda x: x.day)
        return lags_matrix

    def train(self, ts, train_type: ["Boosting", "Linear"] = "linear"):

        feature = self._feature_enrichment(self._make_lags_matrix(ts))

        X = feature.drop(labels=['target'], axis=1)
        y = feature['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.train_ratio
        )

        if train_type.lower() == "boosting":
            self._train_gradient_boosting(X_train, y_train)
        elif train_type.lower() == "linear":
            self._train_linear_regression(X_train, y_train)

        self.score = self.t_model.score(X_test, y_test)

        X['target'] = y
        X.to_csv('all_ts.csv')

        X_test['target'] = y_test
        X_test.to_csv('test_split.csv')

    def _train_linear_regression(self, X_train, y_train):
        # prepared_param = self._prepare()
        self.t_model = LinearRegression(n_jobs=-1)
        self.t_model.fit(X_train, y_train)
        self.is_trained = True

    def _train_gradient_boosting(self, X_train, y_train):
        # prepared_param = self._prepare()
        self.t_model = GradientBoostingRegressor()
        self.t_model.fit(X_train, y_train)
        self.is_trained = True

    def _generate_next_row(self, ts):
        one_raw_data = ts[:-self.num_lags - 1:-1].values.reshape(
            1,
            self.num_lags,
        )
        current_time = ts.index[-1]
        next_time = current_time + self.granularity
        columns = [f'lag_{num_lag}' for num_lag in range(1, self.num_lags + 1)]
        next_row = pd.DataFrame(
            data=one_raw_data,
            columns=columns,
            index=[next_time],
        )
        return self._feature_enrichment(next_row), next_time

    def _parse_timestamp(self, ts, requested_date):
        current_date = ts.index[-1]
        number_of_time_interval = int(
            (requested_date - current_date) / self.granularity,
        )
        if number_of_time_interval > 0:
            return number_of_time_interval

    @staticmethod
    def _is_entry_to_ts(ts, requested_date):
        return requested_date in ts

    def predict_history(self, ts, n_points=1, iso_timestamp=''):
        if not self.is_trained:
            return
        if iso_timestamp:
            requested_date = datetime.fromisoformat(iso_timestamp)
            if self._is_entry_to_ts(ts, requested_date):
                return ts[ts.index == requested_date]

            n_points = self._parse_timestamp(ts, requested_date)
            if not n_points:
                return None

        local_ts = ts
        predicted_ts = pd.Series()
        for _ in range(n_points + 1):
            lags_matrix, next_time = self._generate_next_row(local_ts)
            predicted_value = self.t_model.predict(lags_matrix)
            local_ts = local_ts.append(
                pd.Series(
                    data=predicted_value.item(),
                    index=[next_time],
                ),
            )
            predicted_ts = predicted_ts.append(
                pd.Series(
                    data=predicted_value.item(),
                    index=[next_time],
                ),
            )
        # predicted_ts.plot()
        # self.ts.plot(figsize=(100, 20))
        # plt.show()
        return predicted_ts


if __name__ == '__main__':
    trained_model = TimeSeriesRegressor(num_lags=60)
    trained_model.train(read_json(), 'Linear')
    print(f'The model was train: {trained_model.is_trained}')
    print(
        'This is predict: ',
        trained_model.predict_history(read_json(), n_points=10),
    )
    a_detector = AnomalyDetector().fit(read_json("test_split.csv"),
                                       trained_model)
    print(f'Standard deviation (test_split): {a_detector}')
    a_detector = AnomalyDetector().fit(read_json("all_ts.csv"),
                                       trained_model)
    # finish
    print(f'Standard deviation (all_ts): {a_detector}')
    print('The end')

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# from timeSeriesRegressor import *
