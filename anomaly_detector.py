import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime


class AnomalyDetector(object):
    def __init__(self):
        self.standard_deviation = 0

    def fit(self, trained_model):
        y_pred = trained_model.t_model.predict(
            trained_model.prepared['X_test']
        )
        calaulate_scuare_deviation = pd.DataFrame(
            index=trained_model.prepared['X_test'].index,
        )
        calaulate_scuare_deviation['y_pred'] = y_pred
        calaulate_scuare_deviation['y_test'] = trained_model.prepared['y_test']
        calaulate_scuare_deviation['mean'] = calaulate_scuare_deviation.apply(
            lambda x: x.mean(),
            axis=1,
        )
        calaulate_scuare_deviation[
            'deviation'
        ] = calaulate_scuare_deviation.apply(
            lambda x: x['y_pred'] - x['mean'],
            axis=1
        )
        calaulate_scuare_deviation['square_deviation'] = calaulate_scuare_deviation.apply(
            lambda x: x['deviation'] ** 2,
            axis=1
        )
        dispersion = calaulate_scuare_deviation['square_deviation'].sum() / len(
            calaulate_scuare_deviation['square_deviation'],
        )
        self.standard_deviation = np.sqrt(dispersion)
        return self.standard_deviation

    def detect(self, predict, real_val):
        confidence_intervals = range(
            predict - self.standard_deviation * 3,
            predict + self.standard_deviation * 3,
        )
        return real_val in confidence_intervals
