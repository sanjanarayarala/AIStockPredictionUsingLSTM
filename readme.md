# Stock Price Prediction using LSTM

## Overview

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. It utilizes the Yahoo Finance API to fetch historical stock data, preprocesses it, and trains an LSTM model to make future predictions. The trained model is then deployed using a Streamlit web application for interactive stock price forecasting.

## Project Structure

```
📂 Stock_Price_Prediction_Project
│── stock_price_prediction.ipynb    # Jupyter Notebook for training & saving the LSTM model
│── stock_prediction_app.py         # Streamlit app for making predictions using the trained model
│── stock_lstm.pth                  # Saved LSTM model
│── readme.md                       # Read me file
```

## Key Features

- **Fetch Real-time Data**: Uses Yahoo Finance API to collect stock price data.
- **Data Preprocessing**: Scaling, sequence creation, and train-test splitting.
- **LSTM Model**: A deep learning model trained to predict stock prices.
- **Model Deployment**: A Streamlit web app to interactively predict stock prices.

## Installation & Setup

### 1. Clone the Repository

```sh
 git clone <repo_link>
 cd Stock_Price_Prediction_Project
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed, then run:

```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```sh
streamlit run stock_prediction_app.py
```

## How It Works

### Training (Performed in `stock_price_prediction.ipynb`)

1. Fetches stock price data from Yahoo Finance.
2. Preprocesses the data (scaling, feature engineering, etc.).
3. Trains an LSTM model.
4. Saves the trained model as `stock_lstm.pth`.

### Prediction (Performed in `stock_prediction_app.py`)

1. Accepts user input for stock symbol and date range.
2. Trains & loads the `stock_lstm.pth` model.
3. Fetches recent stock data and processes it.
4. Uses the LSTM model to predict stock prices.
5. Displays predictions through an interactive UI.

## Example Usage

- Enter a stock ticker (e.g., AAPL for Apple) in the Streamlit app.
- Select a date range.
- Click the **Predict** button to see the forecasted stock prices.

## Dependencies

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- yfinance
- Matplotlib
- Streamlit

## Future Improvements

- Incorporating multiple stock features (e.g., trading volume, technical indicators).
- Using more advanced deep learning architectures.
- Enhancing the UI with better visualization tools.

## Author

Developed by **Sanjana**

##

