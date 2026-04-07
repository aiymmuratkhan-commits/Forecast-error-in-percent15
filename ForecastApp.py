import csv


class ForecastApp:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def create_file(self):
        data = [
            ["y_true", "y_pred"],
            [100, 90],
            [200, 210],
            [150, 140],
            [300, 310],
            [250, 240]
        ]

        with open(self.filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def load_data(self):
        with open(self.filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append({
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"])
                })

    def calculate_mape(self):
        total_error = 0
        count = 0

        for row in self.data:
            y_true = row["y_true"]
            y_pred = row["y_pred"]

            if y_true != 0:
                error = abs((y_true - y_pred) / y_true)
                total_error += error
                count += 1

        return (total_error / count) * 100 if count != 0 else 0

    def run(self):
        self.create_file()
        self.load_data()
        mape = self.calculate_mape()
        print(f"Ошибка прогноза (MAPE): {mape:.2f}%")


# 🔹 запуск
if __name__ == "__main__":
    app = ForecastApp("forecast.csv")
    app.run()