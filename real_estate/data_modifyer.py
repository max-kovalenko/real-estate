from sklearn.model_selection import train_test_split

class DataModifyer:
    
    def __init__(self, random_state=None, test_size=0.2, inference=False):
        self.RANDOM_STATE = random_state
        self.test_size = test_size
        self.ctgr_cols = ['city', 'state', 'zip_code']
        self.inference = inference
        
        pass
    
    def modify(self, df):
        df = df \
            .query('status == "for_sale"') \
            .drop(['status', 'prev_sold_date'], axis=1) \
            .dropna()
        df[['city', 'state', 'zip_code']] = \
            df[['city', 'state', 'zip_code']] \
            .apply(lambda x: x.astype('category'))
        df[['bed', 'bath']] = \
            df[['bed', 'bath']] \
            .apply(lambda x: x.astype('Int64'))
        if self.inference: return df
        X_train, X_val, y_train, y_val = train_test_split(
            df.drop('price', axis=1), 
            df['price'], 
            test_size=self.test_size, 
            random_state=self.RANDOM_STATE
        )
        
        return X_train, X_val, y_train, y_val
    
    pass
