import csv


class ForecastProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.y_true = []
        self.y_pred = []

    def create_file(self):
        data = [
            ["y_true", "y_pred"],
            [100, 90],
            [200, 210],
            [0, 50],
            [150, 140],
            [300, 310],
            [0, 10],
            [250, 240]
        ]

        with open(self.filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def load_data(self):
        with open(self.filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.y_true.append(float(row["y_true"]))
                self.y_pred.append(float(row["y_pred"]))

    def filter_non_zero(self):
        filtered_true = []
        filtered_pred = []

        for i in range(len(self.y_true)):
            if self.y_true[i] != 0:
                filtered_true.append(self.y_true[i])
                filtered_pred.append(self.y_pred[i])

        return filtered_true, filtered_pred

    def run(self):
        self.create_file()
        self.load_data()

        y_true_f, y_pred_f = self.filter_non_zero()

        print("После фильтрации:")
        print("y_true:", y_true_f)
        print("y_pred:", y_pred_f)


if __name__ == "__main__":
    app = ForecastProcessor("forecast.csv")
    app.run()