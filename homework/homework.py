#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_Type: Tipo de combustible.
# - Selling_type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import gzip
import json
import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error
)


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, index_col=False, compression="zip")
    df_test = pd.read_csv(test_path, index_col=False, compression="zip")

    print("Datos cargados exitosamente")

    return df_train, df_test


def preprocess_data(df, set_name):    
    # Definir la columna 'Age', asumiendo que el año actual es 2021
    df["Age"] = 2021 - df["Year"]

    # Eliminar las columnas que no son necesarias
    df = df.drop(columns=["Year", "Car_Name"])

    print(f"Preprocesamiento de datos {set_name} completado")

    return df


def split_features_target(df, target_name):
    X = df.drop(columns=[target_name])
    y = df[target_name]

    print("División de características y target completada")

    return X, y


def pipeline_definition(df):
    # Definir las variables categóricas y numéricas
    all_columns = df.columns.tolist()
    categorical_features = [col for col in ["Fuel_Type", "Selling_type", "Transmission"] if col in all_columns]
    numerical_features = [col for col in all_columns if col not in categorical_features]

    print("Variables categóricas: " + str(categorical_features))
    print("Variables numéricas: " + str(numerical_features))

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ]
    )

    # Crear el pipeline
    # 1) one-hot-encoding + numeric scaling (preprocessor)
    # 2) Selección de K mejores
    # 3) Modelo de regresión lineal
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    print("Definición del pipeline completada")

    return pipeline


def hyperparameter_optimization(pipeline, X_train, y_train):
    param_grid = {
        "feature_selection__k": list(range(1, X_train.shape[1] + 1)) + ["all"],
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False],
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    print("Optimización de hiperparámetros completada")

    return grid_search


def save_model(model, model_path):
    # Guardar el modelo comprimido con gzip
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Modelo guardado exitosamente")


def calculate_metrics(model, X, y, dataset_type):
    y_pred = model.predict(X)

    # Calcular métricas
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mad = median_absolute_error(y, y_pred)

    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "r2": r2,
        "mse": mse,
        "mad": mad,
    }

    print(f"Cálculo de métricas completado para el conjunto de {dataset_type}")

    return metrics

def save_metrics(metrics, metrics_path):
    # Guardar las métricas en un archivo json
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

    print("Métricas guardadas exitosamente")


def main():
    df_train, df_test = load_data(
        "files/input/train_data.csv.zip", "files/input/test_data.csv.zip"
    )
    
    df_train = preprocess_data(df_train, "train")
    df_test = preprocess_data(df_test, "test")

    print("Datos preprocesados:")
    print(df_train.head())

    # print(df_train.dtypes)

    X_train, y_train = split_features_target(df_train, "Present_Price")
    X_test, y_test = split_features_target(df_test, "Present_Price")

    pipeline = pipeline_definition(X_train)
    grid_search = hyperparameter_optimization(pipeline, X_train, y_train)

    save_model(grid_search, "files/models/model.pkl.gz")

    train_metrics = calculate_metrics(grid_search, X_train, y_train, "train")
    test_metrics = calculate_metrics(grid_search, X_test, y_test, "test")

    metrics = [train_metrics, test_metrics]
    save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    main()