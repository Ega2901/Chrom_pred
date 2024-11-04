import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Убедитесь, что MLflow отслеживает ваши эксперименты в нужной директории
mlflow.set_experiment('xgboost_regression_experiment')

# Загрузка данных и корректировка названий колонок
data = pd.read_csv('data.csv')

# Удаление ненужной колонки, если она существует
if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)

# Удаление строк с пропущенными значениями
data = data.dropna()

# Корректировка названий колонок сразу после загрузки
data.columns = (
    data.columns.str.replace(r'\[', '_', regex=True)
               .str.replace(r'\]', '_', regex=True)
               .str.replace('<', '', regex=True)
               .str.replace('>', '', regex=True)
               .str.strip()
               .str.replace(' ', '_', regex=True)
)

# Разделение на признаки и целевую переменную
X = data.drop('rt', axis=1)
y = data['rt']

# Преобразование целочисленных колонок с пропусками в float64
for col in X.columns:
    if X[col].dtype == 'int64' and X[col].isnull().any():
        X[col] = X[col].astype('float64')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

mlflow.xgboost.autolog()

# Создание модели XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Запуск эксперимента с MLflow
with mlflow.start_run():
    # Обучение модели
    xgb_model.fit(X_train, y_train)
    
    # Предсказание на тестовой выборке
    predictions = xgb_model.predict(X_test)
    
    # Вычисление средней квадратичной ошибки
    mse = mean_squared_error(y_test, predictions)
    print(f'MSE: {mse}')
    
    # Подготовка примера для сигнатуры
    input_example = X_test.iloc[:5].to_numpy().tolist()  # Пример входных данных
    
    # Сохранение модели с указанием примера входных данных
    mlflow.xgboost.log_model(xgb_model, "model", input_example=input_example)

# Вывод важности признаков
xgb.plot_importance(xgb_model)
plt.show()
