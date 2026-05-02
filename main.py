import ForecastProcessor as forecastprocessor
import ErrorAnalyzer as erroranalyzer
import MAPECalculator11 as mapecalculator
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path


class CsvRepository:
    def load(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден")

        return pd.read_csv(path)


class MAPECalculator:
    @staticmethod
    def calculate(df: pd.DataFrame) -> float:
        df = df[df["y_true"] != 0].copy()

        mape = abs((df["y_true"] - df["y_pred"]) / df["y_true"]).mean()
        return round(mape, 4)

    @staticmethod
    def calculate_by_group(df: pd.DataFrame) -> dict:
        df = df[df["y_true"] != 0].copy()

        result = {}

        for group, data in df.groupby("group"):
            mape = abs((data["y_true"] - data["y_pred"]) / data["y_true"]).mean()
            result[group] = round(mape, 4)

        return result


class JsonRepository:
    def save(self, data: dict, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


class ForecastService:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.csv_repo = CsvRepository()
        self.json_repo = JsonRepository()

    def run(self):
        df = self.csv_repo.load(self.file_path)

        if "group" in df.columns:
            result = {
                "mape": MAPECalculator.calculate(df),
                "groups": MAPECalculator.calculate_by_group(df)
            }
        else:
            result = {
                "mape": MAPECalculator.calculate(df)
            }

        self.json_repo.save(result, "mape_result.json")

        print("\nTASK 14 RESULT:")
        print(result)


class ForecastMAPEAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def _prepare_data(self):
        self.df = self.df[self.df["y_true"] != 0].copy()

    def _calculate_ape(self):
        self.df["ape"] = (
            self.df["y_true"] - self.df["y_pred"]
        ).abs() / self.df["y_true"]

    def calculate_mape(self):
        self._prepare_data()
        self._calculate_ape()

        if "group" in self.df.columns:
            grouped = self.df.groupby("group")["ape"].mean()
            return {k: float(v) for k, v in grouped.items()}

        return {"mape": float(self.df["ape"].mean())}

    def save_to_json(self, result):
        with open("mape_result_old.json", "w") as f:
            json.dump(result, f, indent=4)


class ScatterPlotter:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def run(self):
        plt.figure(figsize=(6, 6))

        plt.scatter(self.y_true, self.y_pred, alpha=0.6)

        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))

        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("Scatter Plot")

        plt.savefig("scatter.png")
        plt.close()

        print("График сақталды: scatter.png")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    file = "forecast.csv"

    # 1
    fp = forecastprocessor.ForecastProcessor(file)
    fp.create_file()
    fp.load_data()

    y_true, y_pred = fp.filter_non_zero()

    print("Filtered Data:")
    print(y_true)
    print(y_pred)

    # 2
    ea = erroranalyzer.ErrorAnalyzer(y_true, y_pred)
    ea.run()

    # 3
    mc = mapecalculator.MAPECalculator(file)
    print("\nOLD MAPE:")
    print(mc.run())

    # 4
    analyzer = ForecastMAPEAnalyzer(file)
    result = analyzer.calculate_mape()

    print("\nNEW MAPE:")
    print(result)

    analyzer.save_to_json(result)

    # 5
    plotter = ScatterPlotter(y_true, y_pred)
    plotter.run()

    # 6 TASK 14
    service = ForecastService(file)
    service.run()