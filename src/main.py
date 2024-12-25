# -*- coding: utf-8 -*-
"""
Transformer Model Trained on Commodities Data

"""

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import requests
import math
import matplotlib.pyplot as plt
import copy
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cut_tree

np.random.seed(42)
warnings.filterwarnings('ignore')

"""**Absolute Price, Log Returns, and Volume of Exxon Mobil**"""

xom = yf.download('XOM', start='2003-01-01')
xom.columns = xom.columns.droplevel('Ticker')
xom.reset_index(inplace=True)
xom['Date'] = pd.to_datetime(xom['Date'])
xom['RollingAdjClose_20'] = xom['Adj Close'].shift(1).rolling(window=20).mean()
xom['Log Returns'] = np.log(xom['Adj Close'] / xom['Adj Close'].shift(1))
xom['Volume 10D Avg'] = xom['Volume'].rolling(window=10).mean()
xom['Realized Volatility'] = xom['Log Returns'].rolling(window=30).std() * (252 ** 0.5)
xom.dropna(inplace=True)
xom = xom.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

"""**RSI**"""

def get_rsi(stock_prices, price_col='Adj Close', window=14):

    delta = stock_prices[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    stock_prices['RSI'] = 100 - (100 / (1 + rs))

    stock_prices = stock_prices.dropna().reset_index(drop=True)

    return stock_prices

"""**MACD & Signal Line**"""

def get_momentum(stock_prices, short_window=12, long_window=26, signal_window=9):

    stock_prices['EMA_12'] = stock_prices['Adj Close'].ewm(span=short_window, adjust=False).mean()
    stock_prices['EMA_26'] = stock_prices['Adj Close'].ewm(span=long_window, adjust=False).mean()

    stock_prices['MACD'] = stock_prices['EMA_12'] - stock_prices['EMA_26']
    stock_prices['Signal Line'] = stock_prices['MACD'].ewm(span=signal_window, adjust=False).mean()

    stock_prices.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True)

    return stock_prices.dropna().reset_index(drop=True)

xom = get_momentum(xom)
xom.to_csv('xom.csv')

"""**Crack Spread**"""

# Get crude data
api_key = "my api key"
url = "https://data.nasdaq.com/api/v3/datatables/QDL/OPEC"

params = {
    "api_key": api_key,
    "date.gte": "2003-01-01",
}

response = requests.get(url, params=params)
data = response.json()
records = data['datatable']['data']
columns = [col['name'] for col in data['datatable']['columns']]

opec_data = pd.DataFrame(records, columns=columns)

# Format crude data
crude_prices = opec_data.rename(columns={'date': 'Date', 'value': 'Crude Price'})
crude_prices['Date'] = pd.to_datetime(crude_prices['Date'])
crude_prices = crude_prices.sort_values(by='Date').reset_index(drop=True)
crude_prices.to_csv('crude_prices.csv', index=False)

# Get reformulated regular gasoline data
url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
params = {
    "frequency": "daily",
    "data[0]": "value",
    "facets[product][]": "EPMRR",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "start": "2003-01-01",
    "offset": 0,
    "length": 5000,
    "api_key": "my api key"
}

all_records = []

while True:
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    records = data['response']['data']
    all_records.extend(records)

    if len(records) < params["length"]:
        break

    params["offset"] += params["length"]

df = pd.DataFrame(all_records)

# Format gas data
gas_prices = df[['period', 'value']].copy()
gas_prices.rename(columns={'period': 'Date', 'value': 'Gas Price'}, inplace=True)
gas_prices['Date'] = pd.to_datetime(gas_prices['Date'])
gas_prices = gas_prices.sort_values(by='Date').reset_index(drop=True)

# Convert to dollar/barrel units
gas_prices['Gas Price'] = pd.to_numeric(gas_prices['Gas Price'], errors='coerce')
gas_prices['Gas Price'] = gas_prices['Gas Price'] * 42
gas_prices.to_csv('gas_prices.csv', index=False)

def get_crack_spread(crude_prices, gas_prices, refining_cost=10):

    data = pd.merge(crude_prices, gas_prices, on='Date', how='inner')

    data['Crack Spread'] = data['Gas Price'] - (data['Crude Price'] + refining_cost)
    data = data[['Date', 'Crack Spread']]

    return data

crack_spread = get_crack_spread(crude_prices, gas_prices)
crack_spread.to_csv('crack_spread.csv', index=False)

"""**Futures Crack Spread**"""

# Get crude futures
crude_futures = yf.download('CL=F', start='2003-03-12')
crude_futures = crude_futures[['Adj Close']].reset_index()
crude_futures.rename(columns={'Adj Close': 'Crude Price'}, inplace=True)
crude_futures.columns = crude_futures.columns.get_level_values(0)
crude_futures.to_csv('crude_futures.csv', index=False)

# Get gas futures
gas_futures = yf.download('RB=F', start='2003-03-12')
gas_futures = gas_futures[['Adj Close']].reset_index()
gas_futures.rename(columns={'Adj Close': 'Gas Price'}, inplace=True)
gas_futures.columns = gas_futures.columns.get_level_values(0)
gas_futures['Gas Price'] *= 42
gas_futures.to_csv('gas_futures.csv', index=False)

futures_crack_spread = get_crack_spread(crude_futures, gas_futures)
futures_crack_spread.rename(columns={'Crack Spread': 'Futures Crack Spread'}, inplace=True)
futures_crack_spread.to_csv('futures_crack_spread.csv', index=False)

"""**Spot Futures Spreads**"""

# Load data
crude_spots = pd.read_csv('crude_prices.csv')
crude_futures = pd.read_csv('crude_futures.csv')
gas_spots = pd.read_csv('gas_prices.csv')
gas_futures = pd.read_csv('gas_futures.csv')

def get_spread(spots, futures, commodity):

    merged_data = pd.merge(spots, futures, on='Date', how='inner', suffixes=(' Spot', ' Futures'))
    spread = f'{commodity} F-S Spread'
    merged_data[spread] = merged_data[f'{commodity} Price Futures'] - merged_data[f'{commodity} Price Spot']

    return merged_data[['Date', spread]]

crude_fs_spread = get_spread(crude_spots, crude_futures, 'Crude')
crude_fs_spread.to_csv('crude_fs_spread.csv', index=False)

gas_fs_spread = get_spread(gas_spots, gas_futures, 'Gas')
gas_fs_spread.to_csv('gas_fs_spread.csv', index=False)

"""**Volatility Spread of Inputs/Outputs**"""

ovx = yf.download('^OVX', start='2003-01-01')
ovx['Implied Volatility'] = ovx['Adj Close'] / 100
ovx = ovx[['Implied Volatility']].reset_index()
ovx.columns = ovx.columns.get_level_values(0)
ovx.to_csv('ovx.csv', index=False)
crude_prices = pd.read_csv('crude_prices.csv')

def get_vol_mismatch(ovx, crude_prices, window=30):

    ovx['Date'] = pd.to_datetime(ovx['Date'])
    crude_prices['Date'] = pd.to_datetime(crude_prices['Date'])

    crude_prices['Log Returns'] = np.log(crude_prices['Crude Price'] / crude_prices['Crude Price'].shift(1))
    crude_prices['Realized Volatility'] = (crude_prices['Log Returns'].rolling(window).std() * (252 ** 0.5)) # Realized needs to be annualized

    merged = pd.merge(ovx, crude_prices[['Date', 'Realized Volatility']], on='Date', how='inner')
    merged['Volatility Mismatch'] = merged['Implied Volatility'] - merged['Realized Volatility']
    merged = merged[['Date', 'Volatility Mismatch']]
    return merged.dropna().reset_index(drop=True)

vol_mismatch = get_vol_mismatch(ovx, crude_prices)
vol_mismatch.to_csv('vol_mismatch.csv', index=False)

"""**Compile features**"""

xom = pd.read_csv('xom.csv')
crack_spread = pd.read_csv('crack_spread.csv')
futures_crack_spread = pd.read_csv('futures_crack_spread.csv')
crude_fs_spread = pd.read_csv('crude_fs_spread.csv')
gas_fs_spread = pd.read_csv('gas_fs_spread.csv')
vol_mismatch = pd.read_csv('vol_mismatch.csv')

features = [
    xom,
    crack_spread,
    futures_crack_spread,
    crude_fs_spread,
    gas_fs_spread,
    vol_mismatch,
]

merged_features = pd.merge(features[0], features[1], on='Date')
for df in features[2:]:
    merged_features = pd.merge(merged_features, df, on='Date', how='inner')

merged_features = merged_features[(merged_features['Date'] <= '2024-01-25')]
merged_features.drop(columns=['Unnamed: 0'], inplace=True)
merged_features.to_csv('merged_features.csv', index=False)

"""**Group features**"""

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cut_tree

features = merged_features.drop(columns=['Date','Adj Close']).copy()

# Get Pearson correlation
pearson_corr = features.corr()

# Get Spearman correlation
spearman_corr = features.corr(method='spearman')

# Get Mutual Information
mi_scores = pd.DataFrame(index=features.columns, columns=features.columns)
for col_x in features.columns:
    for col_y in features.columns:
        mi_scores.loc[col_x, col_y] = mutual_info_regression(
            features[[col_x]], features[col_y], random_state=42
        )[0]

mi_scores = mi_scores.astype(float)

# Hierarchical clustering on Pearson
linkage_matrix = linkage(pearson_corr, method='ward')

num_clusters = 4
clusters = cut_tree(linkage_matrix, n_clusters=num_clusters).flatten()
cluster_mapping = {feature: cluster for feature, cluster in zip(pearson_corr.columns, clusters)}

grouped_features = {cluster: [] for cluster in set(clusters)}
for feature, cluster in cluster_mapping.items():
    grouped_features[cluster].append(feature)

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=pearson_corr.columns, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Pearson Correlation)")
plt.xlabel("Features")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

grouped_features

# Groups
group1 = merged_features[grouped_features[0]].to_csv('group1.csv', index=False)
group2 = merged_features[grouped_features[1]].to_csv('group2.csv', index=False)
group3 = merged_features[grouped_features[2]].to_csv('group3.csv', index=False)
group4 = merged_features[grouped_features[3]].to_csv('group4.csv', index=False)

target = merged_features['Adj Close'].to_csv('target.csv', index=False)

# Compute log percentage change for each feature
def preprocess(X):
    log_pct_change = np.empty_like(X)

    for i in range(X.shape[1]):
        prices = X[:, i]
        prices = np.where(prices <= 0, np.nan, prices)
        prices = pd.Series(prices).interpolate().fillna(method='bfill').fillna(method='ffill').values

        log_prices = np.log(prices)
        pct_change = np.diff(log_prices)

        pct_change = np.insert(pct_change, 0, np.nan)
        log_pct_change[:, i] = pct_change

    log_pct_change = log_pct_change[1:]
    return log_pct_change

# Preprocess each group individually
scaled_group1 = preprocess(group1)
scaled_group2 = preprocess(group2)
scaled_group3 = preprocess(group3)
scaled_group4 = preprocess(group4)

# Preprocess target
scaled_target = preprocess(target.reshape(-1, 1))
scaled_target = scaled_target.flatten()

# Initializes the dataset with input features X, target y, sequence length L, and prediction horizon
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, L, horizon):
        self.X = X
        self.y = y
        self.L = L
        self.horizon = horizon
        self.N = X.shape[0]

    def __len__(self):
        return self.N - self.L - self.horizon + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.L]
        y = self.y[idx + self.L : idx + self.L + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

"""**Tranformer Model Setup**"""

# Constructs an encoder-only Transformer
def make_encoder(d_model=128, nhead=8, num_enc_layers=3, dim_feedforward=512, dropout_rate=0.1, input_dim=3):
    enc_layers = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout_rate
        )
        for _ in range(num_enc_layers)
    ])
    input_proj = nn.Linear(input_dim, d_model)
    output_proj = nn.Linear(d_model, 1)
    return {
        'enc_layers': enc_layers,
        'input_proj': input_proj,
        'output_proj': output_proj
    }

# Forward pass for the encoder-only Transformer
def forward_transformer(x_enc, transformer_dict, device='cpu'):
    enc_layers = transformer_dict['enc_layers']
    input_proj = transformer_dict['input_proj']
    output_proj = transformer_dict['output_proj']

    # Project input features to d_model
    enc_emb = input_proj(x_enc)
    enc_out = enc_emb
    for layer in enc_layers:
        enc_out = layer(enc_out)

    # Use the last time step's embedding for prediction
    predictions = output_proj(enc_out[:, -1, :])
    return predictions.squeeze(-1)

"""**Train Model**"""

# Trains an encoder-only Transformer model.
def train_transformer(X_train, y_train, X_val, y_val, L=60, horizon=5, epochs=12, device='cpu', input_dim=3):
    model_dict = make_encoder(d_model=128, nhead=8, num_enc_layers=3,
                                              dim_feedforward=512, dropout_rate=0.1, input_dim=input_dim)

    params = []
    for k, v in model_dict.items():
        if isinstance(v, nn.Module):
            v.to(device)
            params += list(v.parameters())

    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train, L, horizon)
    val_dataset = TimeSeriesDataset(X_val, y_val, L, horizon)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        for k, v in model_dict.items():
            if isinstance(v, nn.Module):
                v.train()

        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = forward_transformer(batch_x, model_dict, device=device)
            loss = criterion(preds, batch_y.mean(dim=1))  # Forecast mean return
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation Phase
        for k, v in model_dict.items():
            if isinstance(v, nn.Module):
                v.eval()

        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = forward_transformer(batch_x, model_dict, device=device)
                loss = criterion(preds, batch_y.mean(dim=1))
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model_dict)

    return best_model

"""**Whole Training Pipeline**"""

# Runs the entire pipeline
def run_pipeline(group_list, target_array, L=60, horizon=5, n_splits=5):
    # I have a mac, so obviously cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure all groups and target have the same number of samples
    min_length = min([group.shape[0] for group in group_list] + [len(target_array)])
    scaled_group_list = [group[:min_length] for group in group_list]
    scaled_target = target_array[:min_length]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold = 1
    results = []

    for train_val_index, test_index in tscv.split(scaled_target):
        print(f"Starting Fold {fold}/{n_splits}")
        train_ratio = 0.8
        val_size = int(len(train_val_index) * (1 - train_ratio))
        train_indices = train_val_index[:-val_size]
        val_indices = train_val_index[-val_size:]

        X_train_list, X_val_list, X_test_list = [], [], []
        for g in range(len(scaled_group_list)):
            X_train = scaled_group_list[g][train_indices]
            X_val = scaled_group_list[g][val_indices]
            X_test = scaled_group_list[g][test_index]
            X_train_list.append(X_train)
            X_val_list.append(X_val)
            X_test_list.append(X_test)

        y_train = scaled_target[train_indices]
        y_val = scaled_target[val_indices]
        y_test = scaled_target[test_index]

        models = []
        for i in range(len(scaled_group_list)):
            input_dim = scaled_group_list[i].shape[1]
            print(f"  Training model for Group {i+1} with {input_dim} features")
            model_dict = train_transformer(
                X_train_list[i], y_train, X_val_list[i], y_val,
                L=L, horizon=horizon, device=device, input_dim=input_dim
            )
            models.append(model_dict)

        fold_results = {'models': models, 'X_test_list': X_test_list, 'y_test': y_test}
        results.append(fold_results)

        print(f"Completed Fold {fold}/{n_splits}\n")
        fold += 1

    return results

"""**Trading Simulation**"""

# Simulates trading performance based on model forecasts
def simulate_trading_performance(results, initial_capital=100000, trading_horizon=5):
    capital = initial_capital
    trade_data = []

    for fold, fold_result in enumerate(results, start=1):
        models = fold_result['models']
        X_test_list = fold_result['X_test_list']
        y_test = fold_result['y_test']

        test_size = len(y_test)
        num_windows = test_size - trading_horizon + 1
        for t in range(num_windows):
            group_forecasts = []
            for g, model_dict in enumerate(models):
                # Prepare input sequence for the current window
                x_enc_np = X_test_list[g][t : t + trading_horizon]
                x_enc = torch.tensor(x_enc_np, dtype=torch.float32).unsqueeze(0).to(
                    next(model_dict['input_proj'].parameters()).device
                )
                with torch.no_grad():
                    pred = forward_transformer(x_enc, model_dict, device=x_enc.device).item()
                group_forecasts.append(pred)

            # Ensemble forecast: average of all group forecasts
            ensemble_forecast = np.mean(group_forecasts)
            actual_return = y_test[t + trading_horizon - 1]

            # Determine long or short position
            if ensemble_forecast > 0:
                pnl = capital * (np.exp(actual_return) - 1)
            else:
                pnl = capital * (1 - 1 / np.exp(actual_return))

            capital += pnl
            trade_data.append({
                'fold': fold,
                'day': t,
                'forecast': ensemble_forecast,
                'actual_return': actual_return,
                'pnl': pnl,
                'cumulative_capital': capital
            })

    trade_df = pd.DataFrame(trade_data)

    total_trades = len(trade_df)
    average_pnl = trade_df['pnl'].mean()
    total_return = (capital - initial_capital) / initial_capital

    performance_summary = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_trades': total_trades,
        'average_pnl': average_pnl,
        'total_return': total_return
    }

    return performance_summary, trade_df

"""**Feature Weight Analysis**"""

# Visualizes the input projection weights for each feature.
def get_feature_weights(model_dict, group_features):
    input_proj = model_dict['input_proj']
    weights = input_proj.weight.data.cpu().numpy()

    feature_weights = np.mean(np.abs(weights), axis=0)

    weight_df = pd.DataFrame({
        'Feature': group_features,
        'Weight': feature_weights
    })

    weight_df['Weight_Normalized'] = weight_df['Weight'] / weight_df['Weight'].sum()
    weight_df = weight_df.sort_values(by='Weight_Normalized', ascending=False).reset_index(drop=True)

    return weight_df

"""**Results**"""

results = run_pipeline(
    group_list=[scaled_group1, scaled_group2, scaled_group3, scaled_group4],
    target_array=scaled_target,
    L=60,
    horizon=5,
    n_splits=5
)

performance_summary, trade_df = simulate_trading_performance(
    results,
    initial_capital=100000,
    trading_horizon=5
)

performance_summary

trade_df.to_csv('trade_df.csv', index=False)
trade_df = pd.read_csv('trade_df.csv')

groups = [
    ['RollingAdjClose_20', 'Crack Spread', 'Futures Crack Spread'],
    ['Log Returns', 'Gas F-S Spread', 'Volatility Mismatch'],
    ['Volume 10D Avg', 'Realized Volatility', 'Crude F-S Spread'],
    ['RSI', 'MACD', 'Signal Line']
]

# Visualize feature weights for each group using the first fold's models
print("\nVisualizing Feature Weights for Each Group (First Fold):\n")
first_fold_models = results[0]['models']
visualize_feature_weights(first_fold_models, groups)

# Normalize cumulative capital so all folds start at $100,000
normalized_df = trade_df.copy()
normalized_df['cumulative_capital'] = normalized_df.groupby('fold')['cumulative_capital'].transform(
    lambda x: x - x.iloc[0] + 100000
)

# Group by fold and plot
plt.figure(figsize=(12, 7))
for fold, group in normalized_df.groupby('fold'):
    plt.plot(group['day'], group['cumulative_capital'], label=f'Fold {fold}')

plt.xlabel('Day')
plt.ylabel('Capital ($)')
plt.title(' Cumulative Capital Over Time by Fold')
plt.legend()
plt.grid()
plt.show()

# Find the fold with the highest final capital.
def find_best_fold(trade_df):
    final_capital_per_fold = trade_df.groupby('fold')['cumulative_capital'].last().reset_index()

    best_fold_row = final_capital_per_fold.loc[final_capital_per_fold['cumulative_capital'].idxmax()]
    best_fold = best_fold_row['fold']
    best_final_capital = best_fold_row['cumulative_capital']

    print(f"Best Fold: Fold {best_fold} with Final Capital: ${best_final_capital:,.2f}")
    print("\nFinal Capital Across All Folds:")
    print(final_capital_per_fold)

    return best_fold, best_final_capital, final_capital_per_fold

best_fold, best_final_capital, final_capital_per_fold = find_best_fold(trade_df)

# Plot Actual vs. Predicted Returns for a specific fold
def plot_actual_vs_predicted(trade_df, fold_number):

    fold_data = trade_df[trade_df['fold'] == fold_number]
    actual_returns = fold_data['actual_return'].values
    predicted_returns = fold_data['forecast'].values

    indices = np.arange(0, len(actual_returns), 6)
    actual_returns = actual_returns[indices]
    predicted_returns = predicted_returns[indices]

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.plot(actual_returns, label='Actual Returns', color='blue', linewidth=2)
    plt.plot(predicted_returns, label='Predicted Returns', color='red', linestyle='--', linewidth=2)

    plt.title('Actual vs. Predicted Returns (smoothed)', fontsize=16)
    plt.xlabel("Days", fontsize=14)
    plt.ylabel("Return", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(trade_df, 1)

# Dist of daily profits
fold_5 = trade_df[trade_df['fold'] == 5]
plt.figure(figsize=(10, 6))
plt.hist(fold_5['pnl'], bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='black', linestyle='dotted', linewidth=1, label='Zero Line')
plt.xlabel('Daily Profit/Loss ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Profits for Fold 5')
plt.legend()
plt.grid()
plt.show()
