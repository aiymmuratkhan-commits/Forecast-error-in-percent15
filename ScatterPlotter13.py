import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    y_true = [100, 200, 300, 400, 500]
    y_pred = [110, 190, 310, 390, 480]

    plotter = ScatterPlotter(y_true, y_pred)
    plotter.run()