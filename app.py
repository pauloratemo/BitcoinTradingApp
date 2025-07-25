import os
import json
import time
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Config ===
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
DASH_PORT = int(os.getenv("DASH_PORT", 8050))

# === Global Variables ===
latest_price = None
current_position = 0
df = None

# === Binance REST API ===
def fetch_historical(symbol=SYMBOL, interval=INTERVAL, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]

# === Trading Strategies ===
def momentum_strategy(df, window=14, threshold=0.001):
    returns = df['close'].pct_change(window)
    signals = returns.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    signals.iloc[:window] = 0
    return signals

def mean_reversion_strategy(df, window=14, threshold=0.01):
    sma = df['close'].rolling(window).mean()
    diff = (df['close'] - sma) / sma
    signals = diff.apply(lambda x: -1 if x > threshold else (1 if x < -threshold else 0))
    signals.iloc[:window] = 0
    return signals

signals_map = {
    "Momentum": momentum_strategy,
    "Mean Reversion": mean_reversion_strategy
}

# === Backtester ===
def backtest(df, signals, starting_cash=10000, transaction_cost=0.001, slippage=0.001):
    cash = starting_cash
    position = 0
    equity_curve = []

    for date, signal in signals.iteritems():
        price = df.loc[date, 'close']
        if position != 0 and np.sign(position) != signal and signal != 0:
            cost = abs(position) * price * (transaction_cost + slippage)
            cash += position * price - cost
            position = 0
        if position == 0 and signal != 0:
            cost = abs(signal) * price * (transaction_cost + slippage)
            cash -= signal * price + cost
            position = signal
        equity = cash + position * price
        equity_curve.append(equity)

    return pd.Series(equity_curve, index=signals.index), position

# === WebSocket Handler ===
def on_message(ws, message):
    global latest_price
    msg = json.loads(message)
    if 'k' in msg:
        candle = msg['k']
        if candle['x']:
            latest_price = float(candle['c'])

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connected")

def start_ws():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

# === Load Initial Data ===
df = fetch_historical()

# === Dash App ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H2("📈 Bitcoin Multi-Strategy Trading Dashboard"),

    html.Div([
        html.Label("Select Strategy"),
        dcc.Dropdown(id='strategy-dropdown',
                     options=[{'label': k, 'value': k} for k in signals_map.keys()],
                     value='Momentum', clearable=False),

        html.Label("Select Date Range"),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            start_date=df.index.min().date(),
            end_date=df.index.max().date()
        ),

        html.Button('🔁 Retrain Models', id='retrain-btn', n_clicks=0, style={"marginTop": "10px"}),

        html.Div(id='retrain-spinner', style={'display': 'none'}, children=[
            dbc.Spinner(size="sm", color="primary"),
            html.Span(" Retraining in progress... Please wait.")
        ], style={"marginTop": "10px"}),

        html.Pre(id='log-output', style={'whiteSpace': 'pre-wrap', 'height': '150px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '10px', 'marginTop': '10px'}),

        html.H4(id='live-pnl', style={'marginTop': '20px', 'color': 'green', 'fontWeight': 'bold'})
    ], style={'width': '25%', 'float': 'left', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='price-chart'),
        dcc.Interval(id='live-update', interval=10 * 1000, n_intervals=0)
    ], style={'width': '70%', 'float': 'right', 'padding': '20px'}),
])

# === Callbacks ===
@app.callback(
    Output('price-chart', 'figure'),
    Output('live-pnl', 'children'),
    Input('strategy-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('live-update', 'n_intervals')
)
def update_graph_and_live_pnl(strategy, start_date, end_date, n):
    global current_position, latest_price

    filtered_df = df.loc[start_date:end_date].copy()
    signals = signals_map[strategy](filtered_df)
    equity, pos = backtest(filtered_df, signals)
    current_position = pos

    candlestick = go.Candlestick(
        x=filtered_df.index,
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='BTC Price'
    )
    equity_line = go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Equity Curve',
        yaxis='y2'
    )
    fig = go.Figure(data=[candlestick, equity_line])
    fig.update_layout(
        title=f"{strategy} Strategy Backtest",
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(title='Equity', overlaying='y', side='right'),
        xaxis=dict(title='Time')
    )

    if latest_price is not None and not filtered_df.empty:
        last_close = filtered_df['close'].iloc[-1]
        live_pnl_value = current_position * (latest_price - last_close)
        live_pnl_text = f"Live PnL: ${live_pnl_value:,.2f} (Pos: {current_position}, Price: ${latest_price:.2f})"
    else:
        live_pnl_text = "Live PnL: Waiting for live price..."

    return fig, live_pnl_text

@app.callback(
    Output('retrain-spinner', 'style'),
    Output('log-output', 'children'),
    Input('retrain-btn', 'n_clicks'),
    prevent_initial_call=True
)
def retrain_models(n_clicks):
    spinner_style = {'display': 'block'}
    logs = ["Retrain triggered...", "Optimizing parameters..."]
    time.sleep(3)  # Simulate optimization
    logs.append("Retrain completed.")
    spinner_style = {'display': 'none'}
    return spinner_style, "\n".join(logs)

# === Start WebSocket Thread ===
ws_thread = threading.Thread(target=start_ws, daemon=True)
ws_thread.start()

# === Run App ===
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=DASH_PORT, debug=False)
