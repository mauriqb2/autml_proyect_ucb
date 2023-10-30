## Proyecto de Modelos de Regresión para Emisiones de CO2

### Resumen:

Este proyecto utiliza el servicio de Machine Learning de Azure junto con MLflow para entrenar, evaluar y registrar dos modelos de regresión basados en árboles: un árbol de decisión y un bosque aleatorio (Random Forest). Los modelos se entrenan utilizando un conjunto de datos que contiene información sobre emisiones de CO2 de vehículos.

### Contenido:

1.  **Orchestrator-hp.ipynb**: Un notebook que coordina la creación o recuperación de recursos de cómputo en Azure, configura y ejecuta trabajos de entrenamiento y realiza barridos (sweeps) de hiperparámetros para optimizar los modelos.
    
2.  **decissiontree.py**: Script de Python que entrena un modelo de árbol de decisión utilizando scikit-learn y registra métricas y detalles del modelo en MLflow.
    
3.  **randomforest.py**: Script de Python que entrena un modelo de bosque aleatorio utilizando scikit-learn y registra métricas y detalles del modelo en MLflow.
    

### Pasos de ejecución:

1.  **Configuración de Azure ML**:
    
    -   Establecer las credenciales y crear un cliente de ML en Azure.
    -   Configurar o recuperar un clúster de cómputo.
2.  **Entrenamiento de Árbol de Decisión**:
    
    -   Configurar y enviar un trabajo de entrenamiento con parámetros específicos.
    -   Realizar un barrido de hiperparámetros para optimizar el modelo.
3.  **Entrenamiento de Bosque Aleatorio**:
    
    -   Configurar y enviar un trabajo de entrenamiento con parámetros específicos.
    -   Realizar un barrido de hiperparámetros para optimizar el modelo.

### Requisitos:

-   Cuenta de Azure.
-   Librerías: `azure.ai.ml`, `azure.identity`, `sklearn`, `pandas`, `mlflow`.
-   Conjunto de datos con emisiones de CO2 y características relacionadas de vehículos.

### Contribuidores:

-   Mauricio Alejandro Quezada Bustillo
-   Andres Espinoza Soria

## Dataset Link

https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles

### CONTENT

This dataset captures the details of how CO2 emissions by a vehicle can vary with the different features. The dataset has been taken from Canada Government official open data website. This is a compiled version. This contains data over a period of 7 years.  
There are total 7385 rows and 12 columns. There are few abbreviations that has been used to describe the features. I am listing them out here. The same can be found in the Data Description sheet.

### Model

4WD/4X4 = Four-wheel drive  
AWD = All-wheel drive  
FFV = Flexible-fuel vehicle  
SWB = Short wheelbase  
LWB = Long wheelbase  
EWB = Extended wheelbase

### Transmission

A = Automatic  
AM = Automated manual  
AS = Automatic with select shift  
AV = Continuously variable  
M = Manual  
3 - 10 = Number of gears

### Fuel type

X = Regular gasoline  
Z = Premium gasoline  
D = Diesel  
E = Ethanol (E85)  
N = Natural gas

### Fuel Consumption

City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per gallon (mpg)

### CO2 Emissions

The tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving