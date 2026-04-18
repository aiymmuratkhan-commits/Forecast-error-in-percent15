import pandas as pd
import json


class ForecastMAPEAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def _prepare_data(self):
        self.df = self.df[self.df["y_true"] != 0].copy()

    def _calculate_ape(self):
        self.df["ape"] = (self.df["y_true"] - self.df["y_pred"]).abs() / self.df["y_true"]

    def calculate_mape(self):
        self._prepare_data()
        self._calculate_ape()

        if "group" in self.df.columns:
            grouped = self.df.groupby("group")["ape"].mean()
            result = {k: float(v) for k, v in grouped.items()}
        else:
            result = {"mape": float(self.df["ape"].mean())}

        return result

    def save_to_json(self, result, filename="mape_result.json"):
        with open(filename, "w") as f:
            json.dump(result, f, indent=4)

analyzer = ForecastMAPEAnalyzer("forecast.csv")

result = analyzer.calculate_mape()
print(result)

analyzer.save_to_json(result)