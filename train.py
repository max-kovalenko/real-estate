import subprocess

import hydra
import lightgbm as lgb
import mlflow
import pandas as pd
from infer import InferRealEstate
from joblib import dump
from real_estate.data_modifyer import DataModifyer
from real_estate.scaler import MyStandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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

    def train(self, n_estimators, max_depth, learning_rate, num_leaves, mlflow_uri):
        lgbm_regr = lgb.LGBMRegressor(
            random_state=self.RANDOM_STATE,
            n_estimators=n_estimators,
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
        df_val = InferRealEstate().df
        X_val, y_val = df_val.drop("price", axis=1), df_val["price"]
        prediction = lgbm_regr.predict(X_val)
        metrics = {
            "mse": mean_squared_error(y_val, prediction),
            "r2_score": r2_score(y_val, prediction),
        }
        # Logging experiment
        mlflow.set_tracking_uri(uri=mlflow_uri)
        mlflow.set_experiment("RealEstate | LGBM")
        with mlflow.start_run():
            git_commit_id = (
                subprocess.check_output(["git", "log", "--pretty=oneline"])
                .decode()
                .splitlines()[0]
                .split()[0]
            )
            mlflow.log_params(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                    "git_commit_id": git_commit_id,
                }
            )
            feature_importances = dict(
                zip(self.X_train.columns, lgbm_regr.feature_importances_)
            )
            feature_importance_metrics = {
                f"feature_importance_{feature_name}": imp_value
                for feature_name, imp_value in feature_importances.items()
            }
            metrics.update(feature_importance_metrics)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("Training Info", "Basic LGBM model")

    pass


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    rlst = TrainRealEstate()
    rlst.train(**cfg["lgbm"]["train_params"], **cfg["mlflow"]["mlflow_server"])

    pass


if __name__ == "__main__":
    main()
