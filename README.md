# Task 1 – Sales & Demand Forecasting

## Overview
Built a machine learning model to forecast future daily sales using historical business data.  
The model uses time‑based features to predict the next 50 days of sales.

## Approach
- **Data**: Generated synthetic daily sales (500 days) as a proof of concept.
- **Feature Engineering**: day of year, day of week, month.
- **Model**: Linear Regression from Scikit‑learn.
- **Evaluation**: Mean Absolute Error (MAE).

## Results
- MAE on test set: 5.05 *(replace with your actual MAE)*.
- Forecast plot shows a clear trend over the last 50 days.

## Business Insight
- Sales are trending **upward/downward** *(choose one based on your plot)* over the forecast period.
- Average predicted daily sales: **~105.3** *(replace with your actual value)*.
- Recommendation: align inventory/staffing with the predicted demand swing.

## Visual
![Sales Forecast](sales_forecast.png)

## Tools Used
Python, Pandas, NumPy, Scikit‑learn, Matplotlib