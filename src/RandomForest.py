import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

def main():
    # obtener parámetros:
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    
    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()

    lines = [
        f"Data file: {args.dataset_path}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}",
    ]

    print("Parametros: ...")

    # imprimir parámetros:
    for line in lines:
        print(line)

    # log en mlflow
    mlflow.log_param('Data file', str(args.dataset_path))
    mlflow.log_param('Number of Estimators', str(args.n_estimators))
    mlflow.log_param('Max Depth', str(args.max_depth))

    # leer dataset
    data = pd.read_csv(args.dataset_path)
    data = data.dropna()

    # separar el ds
    X = data.drop(columns=['CO2 Emissions(g/km)'])
    X = pd.get_dummies(X, columns=['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'])
    y = data['CO2 Emissions(g/km)']

    # separar el ds en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # entrenar modelo de RandomForest
    rf = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
    rf.fit(X_train, y_train)

    # evaluar modelos
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(r2)

    # imprimir metrica en mlflow
    mlflow.log_metric('R2 Score', float(r2))

    registered_model_name = "sklearn-randomforest-regressor"

    print("Registrar el modelo via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=rf,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    print("Guardar el modelo via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=rf,
        path=os.path.join(registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == '__main__':
    main()

