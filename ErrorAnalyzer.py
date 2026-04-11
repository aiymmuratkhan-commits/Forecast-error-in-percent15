import numpy as np


class ErrorAnalyzer:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def calculate_errors(self):
        errors = np.abs((self.y_true - self.y_pred) / self.y_true) * 100
        return errors

    def mean_error(self):
        return np.mean(self.calculate_errors())

    def run(self):
        errors = self.calculate_errors()
        mean = self.mean_error()

        print("Вектор относительных ошибок (%):", errors)
        print(f"Средняя ошибка: {mean:.2f}%")



if __name__ == "__main__":
    y_true = [100, 200, 150, 300, 250]
    y_pred = [90, 210, 140, 310, 240]

    app = ErrorAnalyzer(y_true, y_pred)
    app.run()