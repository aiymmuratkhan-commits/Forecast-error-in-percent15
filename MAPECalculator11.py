import pandas as pd
import numpy as np


class MAPECalculator:
    def __init__(self, file_path):
        # CSV файл жолы
        self.file_path = file_path
        self.df = None

    def load_data(self):
        # pandas арқылы CSV оқу
        self.df = pd.read_csv(self.file_path)

    def calculate_ape(self):
        # y_true != 0 фильтр (mask)
        mask = self.df["y_true"] != 0

        # Абсолютті проценттік қате (APE) формуласы:
        # |(y_true - y_pred) / y_true|
        self.df.loc[mask, "ape"] = np.abs(
            (self.df.loc[mask, "y_true"] - self.df.loc[mask, "y_pred"])
            / self.df.loc[mask, "y_true"]
        )

        # y_true = 0 болса, есептелмейді
        self.df.loc[~mask, "ape"] = np.nan

    def calculate_mape(self):
        # MAPE = ape бағанының орташа мәні
        return self.df["ape"].mean()

    def run(self):
        # Барлық қадамдарды іске қосу
        self.load_data()
        self.calculate_ape()
        return self.calculate_mape()


# --------- Іске қосу бөлігі ---------
calc = MAPECalculator("forecast.csv")
mape = calc.run()
print("MAPE =", mape)
print(calc.df)