import pandas as pd
import pickle

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

from real_estate.scaler import MyStandardScaler
from real_estate.data_modifyer import DataModifyer

class TrainRealEstate():
    
    def __init__(self, datafile='data/realtor-data.csv', test_size=0.2):
        self.RANDOM_STATE = 42
        self.data_file = datafile
        self.test_size = test_size
        self.df = pd.read_csv(self.data_file)
        mdf = DataModifyer(self.RANDOM_STATE, self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = mdf.modify(self.df)
        scl = MyStandardScaler(file_path='data/scaler.pickle')
        scl.fit(self.X_train)
        self.X_train, self.X_val = map(lambda x: scl.transform(x), [self.X_train, self.X_val])
        
        pass
    
    def train(self, n_estimators=100, max_depth=6, learning_rate=0.1, num_leaves=31):
        lgbm_regr = lgb.LGBMRegressor(
            random_state=self.RANDOM_STATE,
            n_estimators=n_estimators, #100
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves
        )
        lgbm_regr.fit(
            self.X_train, 
            self.y_train, 
            feature_name=['bed', 'bath', 'acre_lot', 'city', 'state', 'zip_code', 'house_size'], 
            categorical_feature=['city', 'state']
        )
        y_pred = lgbm_regr.predict(self.X_val)
        print(f'mean_squared_error: {mean_squared_error(self.y_val, y_pred)}')
        print(f'r2_score: {r2_score(self.y_val, y_pred)}')
        pckl_obj = {'model': lgbm_regr}
        with open('data/lgbm_model.pickle', 'wb') as fp:
            pickle.dump(pckl_obj, fp)
    
    pass

def main(datafile, test_size, n_estimators, max_depth, learning_rate, num_leaves):
    rlst = TrainRealEstate(datafile, test_size)
    rlst.train(n_estimators, max_depth, learning_rate, num_leaves)
    
    pass

if __name__ == '__main__':
    main(
        datafile='data/realtor-data.csv', 
        test_size=0.2, 
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1,
        num_leaves=31
    )
