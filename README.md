[README.md](https://github.com/user-attachments/files/24768206/README.md)
# Inventory Demand Forecasting Model

A predictive analytics tool that uses historical sales data to forecast future inventory needs. This model helps supply chain teams optimize stock levels, reducing both overstock costs and stockout risks.

## Features
- **Data Aggregation:** Automatically aggregates daily sales data into weekly trends.
- **Trend Analysis:** Uses Linear Regression to identify growth or decline in product demand.
- **Visual Reporting:** Generates plots comparing historical data with future forecasts.
- **Actionable Output:** Provides specific unit count predictions for upcoming weeks.

## Tech Stack
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Linear Regression)
- **Visualization:** Matplotlib

## Usage
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Run the model:
   ```bash
   python forecast_model.py
   ```
   *(Note: The script generates mock data automatically for demonstration purposes.)*
