class DataModifyer:
    def __init__(self, inference=False):
        self.ctgr_cols = ["city", "state", "zip_code"]
        self.inference = inference

        pass

    def modify(self, df):
        df = (
            df.query('status == "for_sale"')
            .drop(["status", "prev_sold_date"], axis=1)
            .dropna()
        )
        df[["city", "state", "zip_code"]] = df[["city", "state", "zip_code"]].apply(
            lambda x: x.astype("category")
        )
        df[["bed", "bath"]] = df[["bed", "bath"]].apply(lambda x: x.astype("Int64"))
        if self.inference:
            return df

        return df.drop("price", axis=1), df["price"]

    pass
