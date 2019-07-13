import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime


class AnomalyDetector:
    def __init__(self):
        self.std = 0

    def fit(self, ts, trained_model):
        y_test = ts['target']
        y_pred = trained_model.t_model.predict(
            ts.drop(labels=['target'], axis=1)
        )
        calculate_square_deviation = pd.DataFrame(index=y_test.index)
        calculate_square_deviation['y_pred'] = y_pred
        calculate_square_deviation['y_test'] = y_test
        calculate_square_deviation['mean'] = calculate_square_deviation.apply(
            lambda x: x.mean(),
            axis=1,
        )
        calculate_square_deviation[
            'deviation'] = calculate_square_deviation.apply(
            lambda x: x['y_pred'] - x['mean'],
            axis=1,
        )
        calculate_square_deviation[
            'square_deviation'] = calculate_square_deviation.apply(
            lambda x: x['deviation'] ** 2,
            axis=1,
        )
        dispersion = calculate_square_deviation['square_deviation'].sum() / len(
            calculate_square_deviation['square_deviation'],
        )
        self.std = np.sqrt(dispersion)
        return self.std

    def detect(self, predict, real_val):
        confidence_intervals = range(
            predict - self.std * 3,
            predict + self.std * 3,
        )
        return real_val in confidence_intervals
