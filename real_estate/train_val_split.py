from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter:
    def __init__(self, file, datadir, random_state):
        self.datadir = datadir
        self.file = file
        self.random_state = random_state

        pass

    def split_data(self, train_file, val_file, test_size):
        df = pd.read_csv(Path(self.datadir) / self.file)
        train, val = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=self.random_state
        )
        train.to_csv(Path(self.datadir) / train_file, index=False)
        val.to_csv(Path(self.datadir) / val_file, index=False)

        pass

    pass


def main():
    spl = Splitter(file="realtor-data.csv", datadir="data/", random_state=4221)
    spl.split_data(train_file="train.csv", val_file="val.csv", test_size=0.2)

    pass


if __name__ == "__main__":
    main()
