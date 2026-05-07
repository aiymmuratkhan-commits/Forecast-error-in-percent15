from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import pandas as pd
import io

app = FastAPI()


class CsvRepository:

    @staticmethod
    def calculate(df: pd.DataFrame) -> float:
        df = df[df["y_true"] != 0].copy()

        mape = (
            abs((df["y_true"] - df["y_pred"]) / df["y_true"])
        ).mean()

        return round(mape, 4)

    @staticmethod
    def calculate_by_group(df: pd.DataFrame) -> dict:
        result = {}

        for group, data in df.groupby("group"):
            mape = (
                abs((data["y_true"] - data["y_pred"]) / data["y_true"])
            ).mean()

            result[group] = round(mape, 4)

        return result

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>MAPE Calculator</title>
        </head>
        <body style="font-family: Arial; text-align:center; margin-top:50px;">
            <h1>📊 MAPE Calculator</h1>
            <p>CSV файл жүкте</p>

            <form action="/mape" enctype="multipart/form-data" method="post">
                <input name="file" type="file" />
                <br><br>
                <button type="submit">Есептеу</button>
            </form>
        </body>
    </html>
    """


@app.post("/mape")
async def calculate_mape(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if "group" in df.columns:
        result = {
            "mape": CsvRepository.calculate(df),
            "groups": CsvRepository.calculate_by_group(df)
        }
    else:
        result = {
            "mape": CsvRepository.calculate(df)
        }

    return result
