# 📈 AAPL Stock Time Series Forecasting

A comprehensive time series forecasting project that demonstrates multiple forecasting techniques applied to Apple Inc. (AAPL) stock data. This project compares the performance of ARIMA, Prophet, and Linear Regression models for stock price prediction.

## 🎯 Project Overview

This project provides a complete pipeline for time series forecasting, including:
- **Data Collection**: Automated fetching of AAPL stock data using Yahoo Finance API
- **Exploratory Data Analysis**: Interactive visualizations and statistical analysis
- **Time Series Decomposition**: Seasonal, trend, and residual component analysis
- **Multiple Forecasting Models**: Implementation and comparison of three different approaches
- **Model Evaluation**: Comprehensive performance metrics and visualizations

## 🚀 Features

- **📊 Interactive Visualizations**: Beautiful Plotly charts for data exploration
- **🔍 Time Series Analysis**: Stationarity tests, decomposition, and seasonality detection
- **🤖 Multiple Models**: ARIMA, Facebook Prophet, and Linear Regression implementations
- **📈 Performance Comparison**: Side-by-side model evaluation with multiple metrics
- **🎨 Professional Plots**: Publication-ready visualizations and summary reports
- **⚡ Efficient Code**: Optimized for performance with vectorized operations

## 📁 Project Structure

```
TimeSeriesForecasting/
├── Notebook.ipynb           # Main Jupyter notebook with complete analysis
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TimeSeriesForecasting
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open and run** `Notebook.ipynb`

## 📊 Models Implemented

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Type**: Statistical time series model
- **Best for**: Data with trends and seasonality
- **Parameters**: ARIMA(1,1,1) configuration
- **Strengths**: Well-established, interpretable, handles non-stationary data

### 2. Prophet
- **Type**: Additive forecasting model by Facebook
- **Best for**: Data with strong seasonal patterns
- **Features**: Automatic seasonality detection, holiday effects, robust to missing data
- **Strengths**: Handles multiple seasonalities, easy to use, robust

### 3. Linear Regression
- **Type**: Machine learning approach with time-based features
- **Best for**: Simple trend modeling
- **Features**: Uses timestamp as feature
- **Strengths**: Fast, simple, good baseline model

## 📈 Key Metrics & Evaluation

The project evaluates models using multiple metrics:

| Metric | Description | Best Value |
|--------|-------------|------------|
| **MSE** | Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | Lower is better |
| **R²** | Coefficient of Determination | Higher is better (0-1) |

## 🎨 Visualizations Included

1. **Stock Price Time Series**: Interactive price and volume charts
2. **Time Series Decomposition**: Trend, seasonal, and residual components
3. **Model Forecasts**: Comparison of all model predictions
4. **Performance Metrics**: Bar charts and comparative analysis
5. **Residual Analysis**: Error distribution and patterns
6. **Absolute Percentage Error**: Model accuracy over time

## 📋 Notebook Contents

### Section 1: Import Libraries
- All necessary packages for data analysis, visualization, and modeling

### Section 2: Data Fetching
- Download 3 years of AAPL stock data from Yahoo Finance
- Data validation and basic statistics

### Section 3: Data Exploration
- Interactive price and volume visualizations
- Basic statistical analysis

### Section 4: Time Series Decomposition
- Automatic period detection based on data length
- Multiplicative decomposition into trend, seasonal, and residual components
- Stationarity testing with Augmented Dickey-Fuller test

### Section 5: Data Preprocessing
- Train/test split (80/20)
- Stationarity checks for modeling

### Section 6: ARIMA Model
- Model fitting with ARIMA(1,1,1)
- Forecast generation with confidence intervals
- Performance evaluation

### Section 7: Prophet Model
- Facebook Prophet implementation
- Seasonality and trend analysis
- Component decomposition visualization

### Section 8: Linear Regression
- Time-based feature engineering
- Simple trend modeling
- Performance metrics including R²

### Section 9: Model Comparison
- Comprehensive comparison of all models
- Performance visualization
- Residual analysis
- Key insights and recommendations

## 🔧 Customization Options

### Modify Stock Symbol
```python
ticker = "AAPL"  # Change to any valid stock symbol
```

### Adjust Time Period
```python
start_date = end_date - timedelta(days=1095)  # Modify days for different period
```

### Change Train/Test Split
```python
split_date = aapl.index[int(len(aapl) * 0.8)]  # Modify 0.8 for different split ratio
```

### ARIMA Parameters
```python
arima_model = ARIMA(train_close, order=(1, 1, 1))  # Modify (p,d,q) parameters
```

## 📊 Sample Results

The project typically shows:
- **Dataset**: ~780 observations over 3 years
- **Training**: 80% of data (~620 observations)
- **Testing**: 20% of data (~160 observations)
- **Model Performance**: Varies by market conditions and model selection

Example performance metrics:
```
Model               MSE      MAE     RMSE    MAPE(%)   R² Score
ARIMA (1,1,1)      45.2     5.8     6.7     3.2%      -
Prophet            38.9     5.1     6.2     2.9%      -
Linear Regression  52.1     6.2     7.2     3.5%      0.7845
```

## 🚨 Important Notes

- **Not Financial Advice**: This project is for educational purposes only
- **Market Risk**: Stock prices are inherently unpredictable
- **Model Limitations**: Past performance doesn't guarantee future results
- **Data Quality**: Results depend on data quality and market conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Areas for Enhancement:
- Additional forecasting models (LSTM, XGBoost, etc.)
- Parameter optimization and hyperparameter tuning
- Multiple stock symbols comparison
- Real-time data integration
- Advanced feature engineering
- Model ensemble techniques

## 📝 License

This project is open source and available under the MIT License.

## 📚 References & Resources

- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels ARIMA Guide](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Time Series Analysis Best Practices](https://otexts.com/fpp3/)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Review the notebook comments and documentation
3. Open an issue on the repository

## 🔧 Troubleshooting

### Common Issues:

**1. Prophet Installation Error**:
```bash
pip install prophet
# If fails, try:
conda install -c conda-forge prophet
```

**2. Data Download Issues**:
- Check internet connection
- Verify stock symbol is valid
- Try reducing the date range

**3. Memory Issues**:
- Reduce the dataset size
- Close other applications
- Consider using a more powerful machine

**4. Plotting Issues**:
- Ensure all visualization libraries are installed
- Check Jupyter notebook extensions
- Try restarting the kernel

---

**Made with ❤️ for the data science community**