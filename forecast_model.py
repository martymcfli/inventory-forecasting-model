import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class InventoryForecaster:
    def __init__(self, sales_data_file):
        self.df = pd.read_csv(sales_data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.model = LinearRegression()

    def preprocess_data(self):
        """Aggregate sales by week to smooth out daily volatility."""
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        self.weekly_sales = self.df.groupby('Week')['Units_Sold'].sum().reset_index()
        print("Data aggregated by week.")

    def train_model(self):
        """Train a simple linear regression model on weekly sales."""
        X = self.weekly_sales[['Week']]
        y = self.weekly_sales['Units_Sold']
        
        self.model.fit(X, y)
        print(f"Model trained. Coefficient: {self.model.coef_[0]:.2f}")

    def predict_demand(self, weeks_ahead=4):
        """Predict demand for the next N weeks."""
        last_week = self.weekly_sales['Week'].max()
        future_weeks = np.array([[last_week + i] for i in range(1, weeks_ahead + 1)])
        
        predictions = self.model.predict(future_weeks)
        
        print("\n--- Demand Forecast ---")
        for week, pred in zip(future_weeks.flatten(), predictions):
            print(f"Week {week}: {int(pred)} units expected")
        
        return future_weeks, predictions

    def visualize_forecast(self, future_weeks, predictions):
        """Plot historical sales vs. forecast."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.weekly_sales['Week'], self.weekly_sales['Units_Sold'], color='blue', label='Historical Sales')
        plt.plot(future_weeks, predictions, color='red', linestyle='--', label='Forecast')
        plt.xlabel('Week Number')
        plt.ylabel('Units Sold')
        plt.title('Inventory Demand Forecast')
        plt.legend()
        plt.savefig('forecast_plot.png')
        print("Forecast plot saved to forecast_plot.png")

# Mock Data Creation for Demo
def create_mock_data():
    dates = pd.date_range(start="2023-01-01", periods=90, freq="D")
    sales = np.random.poisson(lam=20, size=90) + np.linspace(0, 10, 90) # Increasing trend
    df = pd.DataFrame({'Date': dates, 'Units_Sold': sales})
    df.to_csv('sales_data.csv', index=False)
    print("Mock sales_data.csv created.")

if __name__ == "__main__":
    create_mock_data()
    
    forecaster = InventoryForecaster('sales_data.csv')
    forecaster.preprocess_data()
    forecaster.train_model()
    future_weeks, predictions = forecaster.predict_demand()
    forecaster.visualize_forecast(future_weeks, predictions)
