# import packages
import importlib
from datetime import datetime, timedelta

from haversine import haversine
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import process

importlib.reload(process)


# define the transformer
class FeatureEngineering():

    def __init__(self, ndays=[1,30], encoder=None, scaler=None):
        self.X = None
        self.ndays = ndays
        self.encoder = encoder
        self.scaler = scaler

    def __calc_aggregate_params_n_days(self, X):
        columns = []
        agg_data = []
        time_index_df = X.set_index('trans_date_trans_time')
        cc_num_df = time_index_df.groupby('cc_num')
        for n in self.ndays:
            cc_num_ndays_df = cc_num_df.rolling(f'{n}D')['amt']
            cc_num_ndays_freq_dict = cc_num_ndays_df.count().to_dict()
            cc_num_ndays_total_dict = cc_num_ndays_df.sum().to_dict()
            cc_num_ndays_freq = [cc_num_ndays_freq_dict[(X.iloc[row]['cc_num'], X.iloc[row]['trans_date_trans_time'])] for row in range(X.shape[0])]
            cc_num_ndays_total = [cc_num_ndays_total_dict[(X.iloc[row]['cc_num'], X.iloc[row]['trans_date_trans_time'])] for row in range(X.shape[0])]
            agg_data += [cc_num_ndays_freq, cc_num_ndays_total]
            columns += [f'card_{n}d_freq', f'card_{n}d_total']

        merchant_df = time_index_df.groupby('merchant')
        for n in self.ndays:
            merchant_ndays_df = merchant_df.rolling(f'{n}D')['amt']
            merchant_ndays_freq_dict = merchant_ndays_df.count().to_dict()
            merchant_ndays_total_dict = merchant_ndays_df.sum().to_dict()
            merchant_ndays_freq = [merchant_ndays_freq_dict[(X.iloc[row]['merchant'], X.iloc[row]['trans_date_trans_time'])] for row in range(X.shape[0])]
            merchant_ndays_total = [merchant_ndays_total_dict[(X.iloc[row]['merchant'], X.iloc[row]['trans_date_trans_time'])] for row in range(X.shape[0])]
            agg_data += [merchant_ndays_freq, merchant_ndays_total]
            columns += [f'merchant_{n}d_freq', f'merchant_{n}d_total']
        
        result = pd.DataFrame(agg_data).T
        result.columns = columns
        return result

    def __transform_transaction_time(self, dt):
        new_dt = datetime.strptime(dt, r'%Y-%m-%d %H:%M:%S')
        return new_dt

    def __is_weekday(self, dt):
        iswday = dt.weekday() < 5
        return iswday

    def __is_daytime(self, dt):
        isday = (dt.hour <= 7) & (dt.hour >= 23)
        return isday

    def __get_age_at_trans(self, record):
        age = (record['trans_date_trans_time'] - datetime.strptime(record['dob'], r'%Y-%m-%d'))
        age_yrs = age//timedelta(days=365.2425)
        return age_yrs

    def __get_distance_between_customer_merchant(self, record):
        coord1 = (record['lat'], record['long'])
        coord2 = (record['lat'], record['long'])
        distance_km = haversine(coord1, coord2)
        return distance_km

    def transform(self, X):
        # Derived Parameters
        X = X.drop(['first', 'last', 'street', 'city', 'zip', 'city_pop',
                    'trans_num', 'unix_time', 'state', 'job'], axis=1)
        X['trans_date_trans_time'] = X['trans_date_trans_time'].apply(self.__transform_transaction_time)
        X['weekday'] = X['trans_date_trans_time'].apply(self.__is_weekday)
        X['hour'] = X['trans_date_trans_time'].apply(self.__is_daytime)
        X['cc_num'] = X['cc_num'].astype(int)
        X['category'] = X['category'].astype(str)
        X['merchant'] = X['merchant'].astype(str)
        X['amt'] = X['amt'].astype(float)
        X['gender'] = X['gender'].astype(str)
        X['age'] = X.apply(self.__get_age_at_trans, axis=1)
        X['distance'] = X.apply(self.__get_distance_between_customer_merchant, axis=1)
        X = X.drop(['lat', 'long', 'merch_lat', 'merch_long', 'dob'], axis=1)
        X = X.sort_values(by=['trans_date_trans_time'])
        X = X.reset_index(drop=True)

        # Aggregated parameters
        agg_params = self.__calc_aggregate_params_n_days(X)
        
        final_X = pd.concat([X, agg_params], axis=1)
        final_X = final_X.drop(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant'], axis=1)
        return final_X
    
    def preprocess(self, X):
        cat_vars = ['gender', 'category']
        # one hot encoding
        if self.encoder is None:
            self.encoder = OneHotEncoder(sparse=False).fit(X[cat_vars])
        cat_x = self.encoder.transform(X[cat_vars])
        # min max scaling
        real_vars = [i for i in X.columns if i not in cat_vars]
        if self.scaler is None:
            self.scaler = MinMaxScaler().fit(X[real_vars])
        real_x = self.scaler.transform(X[real_vars])
        final_x = np.concatenate((cat_x, real_x), axis=1)
        variables = self.encoder.get_feature_names_out(cat_vars).tolist() + real_vars
        return final_x, variables



