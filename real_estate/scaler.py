import os
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class MyStandardScaler(TransformerMixin, BaseEstimator):
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.cols_to_transform = ["bed", "bath", "acre_lot"]

        if (file_path is not None) and (os.path.isfile(self.file_path)):
            with open(self.file_path, "rb") as fp:
                self.scl = pickle.load(fp)["scaler"]
        else:
            self.scl = None

        pass

    def fit(self, df):
        self.scl = StandardScaler()
        self.scl.fit(df[self.cols_to_transform])
        pckl_obj = {"scaler": self.scl}
        if self.file_path is not None:
            with open(self.file_path, "wb") as fp:
                pickle.dump(pckl_obj, fp)

        pass

    def transform(self, df):
        df[self.cols_to_transform] = self.scl.transform(df[self.cols_to_transform])

        return df

    pass
