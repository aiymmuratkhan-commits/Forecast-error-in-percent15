import ForecastProcessor as forecastprocessor
import ErrorAnalyzer as erroranalyzer
import MAPECalculator11 as mapecalculator
import pandas as pd
import json
import matplotlib.pyplot as plt


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


class ScatterPlotter:
    def __init__(self, y_true, y_pred, filename="scatter.png"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.filename = filename

    def plot(self):
        plt.figure(figsize=(6, 6))

        plt.scatter(self.y_true, self.y_pred, alpha=0.6, label="Predictions")

        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal y=x")

        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("Scatter Plot")
        plt.legend()

        plt.savefig(self.filename)
        plt.close()

    def run(self):
        self.plot()
        print(f"График сақталды: {self.filename}")


# ================= MAIN =================
if __name__ == "__main__":
    file = "forecast.csv"

    # 1) ForecastProcessor
    fp = forecastprocessor.ForecastProcessor(file)
    fp.create_file()
    fp.load_data()

    y_true, y_pred = fp.filter_non_zero()

    print("Filtered data:")
    print(y_true)
    print(y_pred)

    # 2) ErrorAnalyzer
    ea = erroranalyzer.ErrorAnalyzer(y_true, y_pred)
    ea.run()

    # 3) Old MAPE
    mc = mapecalculator.MAPECalculator(file)
    mape_old = mc.run()
    print("\nOLD MAPE:", mape_old)

    # 4) New MAPE
    analyzer = ForecastMAPEAnalyzer(file)
    result = analyzer.calculate_mape()

    print("\nNEW MAPE:", result)
    analyzer.save_to_json(result)

    # 5) Scatter Plot
    plotter = ScatterPlotter(y_true, y_pred)
    plotter.run()