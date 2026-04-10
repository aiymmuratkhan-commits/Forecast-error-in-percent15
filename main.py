class ForecastEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path

    def calculate_mape(self):
        errors = []

        with open(self.file_path, 'r') as file:
            next(file)  # header өткізу
            for line in file:
                y_true, y_pred = map(float, line.strip().split(','))
                if y_true != 0:
                    errors.append(abs((y_true - y_pred) / y_true))

        return sum(errors) / len(errors)


evaluator = ForecastEvaluator("forecast.csv")
mape = evaluator.calculate_mape()

print(mape)
