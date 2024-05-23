# Climate-Prediction-Pipeline

Predicting London's climate using machine learning techniques. This project aims to forecast mean temperature in Celsius (°C) using various regression models and logging experiments with MLflow

<strong><em>You can interact with the model here:</em></strong> [Climate Model](https://huggingface.co/spaces/Netcodez/Climate_Prediction_Model)!

# Aim

This project focuses on predicting the climate in London, England, specifically targeting the mean temperature in degrees Celsius (°C). With the increasing importance of weather predictions for businesses in the face of climate change, this project aims to develop a machine learning pipeline using various regression models.

## Dataset

The dataset used for this project is stored in `london_weather.csv` and includes the following columns:
- `date`: Recorded date of measurement (int)
- `cloud_cover`: Cloud cover measurement in oktas (float)
- `sunshine`: Sunshine measurement in hours (hrs) (float)
- `global_radiation`: Irradiance measurement in Watt per square meter (W/m2) (float)
- `max_temp`: Maximum temperature recorded in degrees Celsius (°C) (float)
- `mean_temp`: Target mean temperature in degrees Celsius (°C) (float)
- `min_temp`: Minimum temperature recorded in degrees Celsius (°C) (float)
- `precipitation`: Precipitation measurement in millimeters (mm) (float)
- `pressure`: Pressure measurement in Pascals (Pa) (float)
- `snow_depth`: Snow depth measurement in centimeters (cm) (float)

## Data Preprocessing

The preprocessing pipeline integrates the handling of missing values and data normalisation. This is achieved through the pipeline for SimpleImputer(replacing the missing values with mean values of each column) and StandardScaler respectively, ensuring seamless data preprocessing before model training.

## Model Evaluation

Metric- Root Mean Square Error (RMSE)

After experimenting with various regression models: 
- Linear Regression,
- Decision Tree Regressor,
- RandomForest Regressor,
the Random Regressor was found to yield the best result with an `RMSE` of **0.861**.

## Transition Model to Testing and Production Stages

Version 3 of the Random Regressor Model was moved to Production stage.

## Result

The Production Model achieved an accuracy of 97.7%.

## Requirements

- Python 3.x
- mlflow 2.10.2
- scikit-learn
- pandas
- numpy
- matplotlib
