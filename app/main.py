from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from model_service import ModelService
import shutil
import os
from pathlib import Path

#создаем приложение и html шаблон
app = FastAPI(title="Heart Attack Risk Prediction API")
templates = Jinja2Templates(directory="templates")

#создаем объект с моделью при старте сервера
model_service = ModelService("best_model.joblib")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    #отображение главной страницы с формой загрузки файла
    return templates.TemplateResponse("upload.html", {"request": request})


#загрузка CSV-файла
@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    #сохраняем временный входной файл
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    #получаем предсказания
    predictions_df = model_service.predictions(temp_path)
    #создаем папку для результатов
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(exist_ok=True)
    #сохраняем CSV с предсказаниями
    output_path = predictions_dir / "predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    #удаляем временный входной файл
    os.remove(temp_path)
    #для HTML преобразуем в список словарей
    predictions = predictions_df.to_dict(orient="records")
    #возвращаем HTML с таблицей предсказаний
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "predictions": predictions}
    )


@app.get("/download")
def predictions_download():
    return FileResponse(
        path="predictions\predictions.csv",
        filename="heart_attack_risk_prediction.csv",
        media_type="text/csv"
    )