import pandas as pd
import pickle

from real_estate.scaler import MyStandardScaler
from real_estate.data_modifyer import DataModifyer

class InferRealEstate():
    
    def __init__(self, datafile='data/realtor-data.csv', modelfile='data/lgbm_model.pickle'):
        self.data_file = datafile
        self.modelfile = modelfile
        self.df = pd.read_csv(self.data_file)
        mdf = DataModifyer(inference=True)
        self.df = mdf.modify(self.df)
        scl = MyStandardScaler(file_path='data/scaler.pickle')
        self.df = scl.transform(self.df)
        
        pass
    
    def inference(self):
        with open(self.modelfile, 'rb') as fp:
            lgbm_regr = pickle.load(fp)['model']
        y_pred = lgbm_regr.predict(self.df.drop('price', axis=1))
        pd.DataFrame({'prediction': y_pred}).to_csv('data/prediction.csv')
        
        pass

    pass

def main(datafile, modelfile):
    rlst = InferRealEstate(datafile, modelfile)
    rlst.inference()
    
    pass
    
if __name__ == '__main__':
    main(
        datafile='data/realtor-data.csv', 
        modelfile='data/lgbm_model.pickle'
    )

