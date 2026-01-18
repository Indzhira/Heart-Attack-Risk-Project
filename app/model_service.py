from joblib import load
import pandas as pd

class ModelService:
    def __init__(self, model_path: str):
        self.model = load(model_path)
        self.drop_cols = ['unnamed:_0', 'ck-mb', 'troponin']
        self.cat_cols = ['diabetes', 'family_history', 'smoking', 'obesity',
                         'alcohol_consumption', 'previous_heart_problems', 'medication_use']
        self.int_cols = ['stress_level', 'physical_activity_days_per_week']
    
    # Предобработка df
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Нормализация названий колонок
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Кодирование столбца gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].replace({ '1.0': 'Male', '0.0': 'Female'})
        
        # Удаление ненужных колонок
        df = df.drop(columns=[col for col in self.drop_cols if col in df.columns])

        # Установка id как индекса (если есть)
        if 'id' in df.columns:
            df = df.set_index('id')
     
        # Обработка категориальных колонок
        new_cat_cols = [col for col in self.cat_cols if col in df.columns]
        if new_cat_cols:
            mode_values = df[new_cat_cols].mode().iloc[0]
            df[new_cat_cols] = df[new_cat_cols].fillna(mode_values)
            df[new_cat_cols] = df[new_cat_cols].astype(int)
        
        # Обработка числовых колонок
        new_int_cols = [col for col in self.int_cols if col in df.columns]
        if new_int_cols:
            median_values = df[new_int_cols].median()
            df[new_int_cols] = df[new_int_cols].fillna(median_values)
            df[new_int_cols] = df[new_int_cols].astype(int)  
        return df
    
    # Получение предсказаний из cvs
    def predictions(self, csv_path: str) -> pd.DataFrame:
        data = pd.read_csv(csv_path)
        data = self.preprocess(data)  
        predictions = pd.DataFrame({
            "id": data.index.tolist(),
            "prediction": self.model.predict(data).astype(int)
        })
        return predictions
