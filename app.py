import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import date, timedelta

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Stock Backtester", page_icon="📈", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0aec0;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        text-align: center;
        padding: 1rem 0;
        font-weight: 700;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }
    div.block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📈 AI-Powered Stock Backtesting Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Professional-grade backtesting simplified for personal use</p>", unsafe_allow_html=True)

# ==========================================
# FUNCTIONS
# ==========================================

@st.cache_data(show_spinner=False)
def load_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

def ma_strategy(df):
    """
    Moving Average Crossover Strategy (50 MA and 200 MA).
    Signal: 1 for UPTREND, -1 for DOWNTREND.
    """
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    df['Signal'] = 0
    df['Signal'] = np.where(df['MA50'] > df['MA200'], 1, -1)
    
    # We only take trade signals on crossing
    df['Trade_Signal'] = df['Signal'].diff()
    return df

def rsi_strategy(df):
    """
    RSI Strategy (RSI 14).
    Buy below 30, Sell above 70.
    """
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    df['Signal'] = 0
    df.loc[df['RSI'] < 30, 'Signal'] = 1
    df.loc[df['RSI'] > 70, 'Signal'] = -1
    return df

def triple_sma_strategy(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = 0
    buy_cond = (df['SMA200'] > df['SMA50']) & (df['SMA50'] > df['SMA20']) & (df['Close'] < df['SMA20']) & (df['Close'] < df['SMA50']) & (df['Close'] < df['SMA200'])
    sell_cond = (df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA200']) & (df['Close'] > df['SMA20']) & (df['Close'] > df['SMA50']) & (df['Close'] > df['SMA200'])
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def knoxville_divergence_strategy(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=21).rsi()
    df['Momentum'] = ta.momentum.ROCIndicator(close=df['Close'], window=20).roc()
    df['Signal'] = 0
    buy_cond = (df['RSI'].shift(1) < 30) & (df['RSI'] > 30) & (df['Momentum'] > 0)
    sell_cond = (df['RSI'].shift(1) > 70) & (df['RSI'] < 70) & (df['Momentum'] < 0)
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def v20_strategy(df):
    df['Ret20'] = df['Close'].pct_change(periods=20)
    df['Max20'] = df['High'].rolling(20).max()
    df['Min20'] = df['Low'].rolling(20).min()
    df['Signal'] = 0
    df['V20_Setup'] = df['Ret20'] >= 0.20
    lower_range = df['Min20'] + 0.3 * (df['Max20'] - df['Min20'])
    upper_range = df['Min20'] + 0.8 * (df['Max20'] - df['Min20'])
    buy_cond = df['V20_Setup'].rolling(30).max() == 1
    buy_cond = buy_cond & (df['Close'] <= lower_range)
    sell_cond = df['Close'] >= upper_range
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def rhs_strategy(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Local_High_30'] = df['High'].rolling(window=30).max()
    df['Signal'] = 0
    buy_cond = (df['Close'] > df['Local_High_30'].shift(1)) & (df['Close'] < df['SMA200'])
    sell_cond = (df['Close'] >= df['High'].cummax().shift(1))
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def cwh_strategy(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Local_High_90'] = df['High'].rolling(window=90).max()
    df['Signal'] = 0
    buy_cond = (df['Close'] > df['Local_High_90'].shift(1)) & (df['Close'] > df['SMA200'])
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    sell_cond = df['Close'] < df['SMA50']
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def v10_strategy(df):
    df['Local_High_30'] = df['High'].rolling(window=30).max()
    df['Signal'] = 0
    buy_cond = df['Close'] <= df['Local_High_30'] * 0.90
    sell_cond = df['Close'] >= df['Local_High_30'].shift(1)
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def thrice_in_three_strategy(df):
    df['Lifetime_High'] = df['High'].cummax()
    df['Signal'] = 0
    buy_cond = df['Close'] <= df['Lifetime_High'] * 0.33
    sell_cond = df['Close'] >= df['Lifetime_High'] * 0.95
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def lifetime_high_strategy(df):
    df['Lifetime_High'] = df['High'].cummax()
    df['Signal'] = 0
    buy_cond = (df['Close'] <= df['Lifetime_High'] * 0.70) & (df['Close'].shift(1) > df['Lifetime_High'].shift(1) * 0.70)
    sell_cond = df['Close'] >= df['Lifetime_High'] * 0.98
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

def backtest(df, initial_capital, risk_per_trade_pct, stop_loss_pct, brokerage_pct, target_profit_pct=None):
    """
    Simulate trades based on signals and calculate performance metrics.
    """
    capital = initial_capital
    position = 0 # Number of shares holding
    buy_price = 0
    
    trade_list = []
    equity_curve = []
    peak_capital = initial_capital
    drawdowns = []
    
    for i in range(len(df)):
        current_price = float(df['Close'].iloc[i])
        date_idx = df.index[i]
        signal = df['Signal'].iloc[i]
        
        # Track if we executed a trade this bar
        trade_executed = False
        
        # 0. Check Target Profit Pre-emptively
        if position > 0 and target_profit_pct is not None:
            if current_price >= buy_price * (1 + target_profit_pct / 100):
                # TARGET PROFIT HIT
                sell_value = position * current_price
                brokerage = sell_value * (brokerage_pct / 100)
                net_sell_value = sell_value - brokerage
                capital += net_sell_value
                trade_list.append({
                    'Date': date_idx, 'Type': 'TARGET HIT', 'Price': current_price, 
                    'Shares': position, 'Capital': capital, 'Return %': ((current_price - buy_price)/buy_price)*100
                })
                position = 0
                buy_price = 0
                trade_executed = True

        # 1. Check Stop Loss
        if not trade_executed and position > 0:
            if current_price <= buy_price * (1 - stop_loss_pct / 100):
                # STOP LOSS HIT
                sell_value = position * current_price
                brokerage = sell_value * (brokerage_pct / 100)
                net_sell_value = sell_value - brokerage
                
                capital += net_sell_value
                trade_list.append({
                    'Date': date_idx, 
                    'Type': 'STOP LOSS', 
                    'Price': current_price, 
                    'Shares': position, 
                    'Capital': capital,
                    'Return %': ((current_price - buy_price)/buy_price)*100
                })
                
                position = 0
                buy_price = 0
                trade_executed = True
                
        # 2. Check Strategy Signals
        if not trade_executed:
            if position == 0 and signal == 1: 
                # BUY SIGNAL
                risk_amount = capital * (risk_per_trade_pct / 100)
                distance_to_sl = current_price * (stop_loss_pct / 100)
                
                if distance_to_sl > 0:
                    shares_to_buy = int(risk_amount / distance_to_sl)
                else:
                    shares_to_buy = 0
                    
                buy_value = shares_to_buy * current_price
                brokerage = buy_value * (brokerage_pct / 100)
                
                # Check if enough capital
                if shares_to_buy > 0 and capital >= (buy_value + brokerage):
                    capital -= (buy_value + brokerage)
                    position = shares_to_buy
                    buy_price = current_price
                    trade_list.append({
                        'Date': date_idx, 
                        'Type': 'BUY', 
                        'Price': current_price, 
                        'Shares': shares_to_buy, 
                        'Capital': capital,
                        'Return %': 0.0
                    })
                    
            elif position > 0 and signal == -1:
                # SELL SIGNAL
                sell_value = position * current_price
                brokerage = sell_value * (brokerage_pct / 100)
                net_sell_value = sell_value - brokerage
                
                capital += net_sell_value
                trade_list.append({
                    'Date': date_idx, 
                    'Type': 'SELL', 
                    'Price': current_price, 
                    'Shares': position, 
                    'Capital': capital,
                    'Return %': ((current_price - buy_price)/buy_price)*100
                })
                position = 0
                buy_price = 0

        # Update Equity Curve & Drawdown
        current_equity = capital
        if position > 0:
            # Mark to market
            current_equity += position * current_price
            
        equity_curve.append(current_equity)
        
        if current_equity > peak_capital:
            peak_capital = current_equity
            
        drawdown = ((peak_capital - current_equity) / peak_capital) * 100
        drawdowns.append(drawdown)
        
    df['Equity'] = equity_curve
    df['Drawdown'] = drawdowns
    
    # Sell remaining positions at the end of the data to close metrics
    if position > 0:
        current_price = float(df['Close'].iloc[-1])
        sell_value = position * current_price
        brokerage = sell_value * (brokerage_pct / 100)
        capital += (sell_value - brokerage)
        trade_list.append({
            'Date': df.index[-1], 
            'Type': 'EOD CLOSE', 
            'Price': current_price, 
            'Shares': position, 
            'Capital': capital,
            'Return %': ((current_price - buy_price)/buy_price)*100
        })
        # Note: we don't append to equity curve here as it's already recorded
        
    final_capital = df['Equity'].iloc[-1] if not df.empty else initial_capital
    profit_loss = final_capital - initial_capital
    
    # Calculate Win Rate from trades
    trade_df = pd.DataFrame(trade_list)
    wins = 0
    total_closed = 0
    
    if not trade_df.empty:
        closed_trades = trade_df[trade_df['Type'].isin(['SELL', 'STOP LOSS', 'EOD CLOSE', 'TARGET HIT'])]
        total_closed = len(closed_trades)
        wins = len(closed_trades[closed_trades['Return %'] > 0])
            
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    
    results = {
        'Final Capital': final_capital,
        'Profit / Loss': profit_loss,
        'Win Rate (%)': win_rate,
        'Total Trades': total_closed,
        'Max Drawdown (%)': max(drawdowns) if drawdowns else 0.0,
        'Trade List': trade_df
    }
    
    return df, results


# ==========================================
# UI LAYOUT
# ==========================================

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
        
    st.divider()
    
    st.subheader("💰 Capital & Risk")
    initial_capital = st.number_input("Initial Capital (₹)", min_value=1000, value=100000, step=1000)
    risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    stop_loss = st.slider("Stop Loss (%)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    brokerage = st.number_input("Brokerage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    
    st.divider()
    
    st.subheader("📊 Strategy Selection")
    strategy_choice = st.selectbox(
        "Choose Strategy", 
        (
            "Triple SMA (200>50>20)", 
            "Knoxville Divergence (Proxy)",
            "V20 Strategy",
            "Reverse Head & Shoulder (Proxy)",
            "Cup & Handle (Breakout)",
            "V10 Strategy (10% Drop)",
            "3x in 3 Years (67% Drop)",
            "Lifetime High (30% Drop)",
            "Moving Average Crossover (50/200)", 
            "RSI Strategy (14, 30/70)"
        )
    )
    
    run_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

# Main View
if run_button:
    with st.spinner("Fetching data and running backtest..."):
        # 1. Load Data
        df = load_data(ticker, start_date, end_date)
        
        if df is None or len(df) < 200:
            st.error(f"Failed to fetch sufficient data for {ticker}. Please check the ticker symbol or try a longer date range (need at least 200 days for MA strategy).")
        else:
            # 2. Apply Strategy Indicators
            target_profit_override = None

            if strategy_choice.startswith("Triple SMA"):
                df = triple_sma_strategy(df)
            elif strategy_choice.startswith("Knoxville Divergence"):
                df = knoxville_divergence_strategy(df)
            elif strategy_choice.startswith("V20"):
                df = v20_strategy(df)
            elif strategy_choice.startswith("Reverse Head & Shoulder"):
                df = rhs_strategy(df)
            elif strategy_choice.startswith("Cup & Handle"):
                df = cwh_strategy(df)
            elif strategy_choice.startswith("V10"):
                df = v10_strategy(df)
            elif strategy_choice.startswith("3x in 3 Years"):
                df = thrice_in_three_strategy(df)
                target_profit_override = 100.0 # 100% gain target
            elif strategy_choice.startswith("Lifetime High"):
                df = lifetime_high_strategy(df)
            elif strategy_choice.startswith("Moving Average"):
                df = ma_strategy(df)
            else:
                df = rsi_strategy(df)
                
            # 3. Run Backtest Engine
            df, results = backtest(df, initial_capital, risk_per_trade, stop_loss, brokerage, target_profit_override)
            
            # 4. Display Dashboard Metrics
            st.markdown("### 🏆 Performance Summary")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            profit_color = "normal" if results['Profit / Loss'] >= 0 else "inverse"
            m1.metric("Final Capital", f"₹{results['Final Capital']:,.2f}", f"{results['Profit / Loss']:,.2f}", delta_color=profit_color)
            m2.metric("Total Profit / Loss", f"₹{results['Profit / Loss']:,.2f}")
            m3.metric("Win Rate", f"{results['Win Rate (%)']:.1f}%")
            m4.metric("Total Trades", results['Total Trades'])
            m5.metric("Max Drawdown", f"{results['Max Drawdown (%)']:.2f}%")
            
            st.divider()
            
            # 5. Visualizations
            st.markdown("### 📈 Interactive Charts")
            
            tab1, tab2, tab3 = st.tabs(["Stock Price & Signals", "Equity Curve", "Drawdown"])
            
            with tab1:
                fig_price = make_subplots(rows=1, cols=1, shared_xaxes=True)
                fig_price.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Price'))
                                    
                if strategy_choice.startswith("Moving Average"):
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA 50', line=dict(color='blue')))
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA 200', line=dict(color='orange')))
                    
                # Mark Buy/Sell signals based on actual executed trades!
                trade_df = results['Trade List']
                if not trade_df.empty:
                    buys = trade_df[trade_df['Type'] == 'BUY']
                    sells = trade_df[trade_df['Type'].isin(['SELL', 'STOP LOSS', 'EOD CLOSE', 'TARGET HIT'])]
                    
                    fig_price.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', 
                                                   marker=dict(symbol='triangle-up', color='green', size=12),
                                                   name='Buy Signal'))
                                                   
                    fig_price.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', 
                                                   marker=dict(symbol='triangle-down', color='red', size=12),
                                                   name='Sell/SL Signal'))
                                                   
                fig_price.update_layout(height=600, template="plotly_dark", title=f"{ticker} Price Chart")
                fig_price.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig_price, use_container_width=True)
                
                if strategy_choice.startswith("RSI"):
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(height=300, template="plotly_dark", title="RSI Indicator (14)")
                    st.plotly_chart(fig_rsi, use_container_width=True)

            with tab2:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=df.index, y=df['Equity'], mode='lines', name='Equity Curve', line=dict(color='lime')))
                fig_eq.update_layout(height=500, template="plotly_dark", title="Portfolio Equity Curve")
                st.plotly_chart(fig_eq, use_container_width=True)
                
            with tab3:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], mode='lines', name='Drawdown %', line=dict(color='red')))
                fig_dd.update_layout(height=500, template="plotly_dark", title="Drawdown (%)", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()
            
            # 6. Trade Log
            st.markdown("### 📋 Trade Log")
            if not trade_df.empty:
                # Format the trade dataframe for display
                display_df = trade_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df['Price'] = display_df['Price'].apply(lambda x: f"₹{x:.2f}")
                display_df['Capital'] = display_df['Capital'].apply(lambda x: f"₹{x:.2f}")
                display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%" if x != 0 else "-")
                
                # Apply styling function to color rows based on profit/loss
                def style_logic(row):
                    if row['Type'] == 'BUY':
                        return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                    elif row['Type'] == 'STOP LOSS':
                        return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                    elif row['Type'] == 'TARGET HIT':
                        return ['background-color: rgba(0, 255, 150, 0.2)'] * len(row)
                    elif row['Type'] in ['SELL', 'EOD CLOSE']:
                        if float(row['Return %'].replace('%','')) > 0:
                            return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
                        else:
                            return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                    return [''] * len(row)

                st.dataframe(display_df.style.apply(style_logic, axis=1), use_container_width=True, hide_index=True)
            else:
                st.info("No trades were executed during this period with the current strategy & parameters.")
else:
    st.info("👈 Please select your configuration from the sidebar and click **Run Backtest** to see the results.")
