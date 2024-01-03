import hydra
import lightgbm as lgb
import pandas as pd
from joblib import dump
from real_estate.data_modifyer import DataModifyer
from real_estate.scaler import MyStandardScaler


class TrainRealEstate:
    def __init__(self, datafile="data/train.csv"):
        self.RANDOM_STATE = 42
        self.data_file = datafile
        self.df = pd.read_csv(self.data_file)
        mdf = DataModifyer()
        self.X_train, self.y_train = mdf.modify(self.df)
        scl = MyStandardScaler(file_path="data/scaler.joblib")
        scl.fit(self.X_train)
        self.X_train = scl.transform(self.X_train)

        pass

    def train(self, n_estimators, max_depth, learning_rate, num_leaves):
        lgbm_regr = lgb.LGBMRegressor(
            random_state=self.RANDOM_STATE,
            n_estimators=n_estimators,  # 100
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )
        lgbm_regr.fit(
            self.X_train,
            self.y_train,
            feature_name=[
                "bed",
                "bath",
                "acre_lot",
                "city",
                "state",
                "zip_code",
                "house_size",
            ],
            categorical_feature=["city", "state"],
        )
        dump({"model": lgbm_regr}, "data/lgbm_model.joblib")

    pass


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    rlst = TrainRealEstate()
    rlst.train(**cfg["lgbm"].items().__iter__().__next__()[1])

    pass


if __name__ == "__main__":
    main()
