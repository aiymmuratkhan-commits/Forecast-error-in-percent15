import pandas as pd
import json
from pathlib import Path

class CsvRepository:
    """Класс для загрузки CSV файла"""

    def load(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден")

        return pd.read_csv(path)

class MAPECalculator:
    """Класс для расчета MAPE"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> float:
        df = df[df["y_true"] != 0].copy()

        mape = (abs((df["y_true"] - df["y_pred"]) / df["y_true"])).mean()
        return round(mape, 4)

    @staticmethod
    def calculate_by_group(df: pd.DataFrame) -> dict:
        df = df[df["y_true"] != 0].copy()

        result = {}

        for group, data in df.groupby("group"):
            mape = (abs((data["y_true"] - data["y_pred"]) / data["y_true"])).mean()
            result[group] = round(mape, 4)

        return result


class JsonRepository:
    """Класс для сохранения JSON"""

    def save(self, data: dict, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


class ForecastService:
    """Главный сервис"""

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

        print("Результат сохранён в mape_result.json")
        print(result)

service = ForecastService("forecast.csv")
service.run()