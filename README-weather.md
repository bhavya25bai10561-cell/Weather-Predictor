# Weather-Predictor
This Python script demonstrates a complete workflow for training a linear regression model to predict temperature based on synthetic weather data, using popular data science libraries like NumPy for data generation, Pandas for data handling and Scikit-learn for the machine learning model.

This Python code implements a Multiple Linear Regression model to predict temperature based on three weather features: humidity, pressure, and wind speed. It demonstrates the complete machine learning workflow, from data generation to model evaluation and visualization. 

Workflow Breakdown
Data Generation: The code uses NumPy to create a synthetic dataset of 1,000 samples. It defines a linear relationship for temperat and adds Gaussian noise to simulate real-world variability.

Data Preparation:
DataFrame: The data is organized using Pandas.
Splitting: The dataset is divided into training (80%) and testing (20%) sets using Scikit-learn's train_test_split to ensure the model's performance is validated on unseen data.
Modeling: A LinearRegression model is trained on the features (x_train) to learn the coefficients that best map inputs to the target temperature (y_train).
Evaluation: The model's accuracy is measured using two standard metrics from Scikit-learn's metrics module:
Mean Absolute Error (MAE): The average magnitude of errors in predictions.
Mean Squared Error (MSE): Penalizes larger errors more heavily by squaring the differences.
Visualization: Using Matplotlib, the code generates four plots to analyze feature correlations and compare actual versus predicted temperature values.
Prediction: Finally, the trained model is used to predict the temperature for a specific new weather scenario.

Multiple Linear Regression is highly effective for predicting temperature when there is a clear, linear relationship between the inputs (humidity, pressure, wind speed) and the output.

