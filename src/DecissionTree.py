from sklearnex import patch_sklearn
patch_sklearn()

import argparse
import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

def main():
    # obtener parámetros:
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=None)

    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()

    lines = [
        f"Data file: {args.dataset_path}",
        f"Min Samples split: {args.min_samples_split}",
        f"Max Depth: {args.max_depth}",
    ]

    print("Parametros: ...")

    # imprimir parámetros:
    for line in lines:
        print(line)

    # log en mlflow
    mlflow.log_param('Data file', str(args.dataset_path))
    mlflow.log_param('Min Samples split', str(args.min_samples_split))
    mlflow.log_param('Max Depth', str(args.max_depth))

    # leer dataset
    data = pd.read_csv(args.dataset_path)

    # separar el ds
    X = data.drop(columns=['CO2 Emissions(g/km)'])
    X = pd.get_dummies(X, columns=['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'])
    y = data['CO2 Emissions(g/km)']

    # separar el ds en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # entrenar modelo de árbol de decisión
    dt = DecisionTreeRegressor(min_samples_split=args.min_samples_split, max_depth=args.max_depth)
    dt.fit(X_train, y_train)

    # evaluar modelos
    y_pred = dt.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # imprimir metrica en mlflow
    mlflow.log_metric('R2 Score', float(r2))

    registered_model_name="sklearn-decision-tree-regressor"

    print("Registrar el modelo via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=dt,  # Change here
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    print("Guardar el modelo via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=dt,  # Change here
        path=os.path.join(registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == '__main__':
    main()



