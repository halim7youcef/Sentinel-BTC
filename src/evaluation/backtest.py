import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRED_DIR = "results/predictions"
DATA_PATH = "data/processed/btc_features_ml.csv"
RESULT_DIR = "results/backtest"

os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading Data baseline for SOTA Backtest...")
data = pd.read_csv(DATA_PATH)
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

prediction_files = glob.glob(f"{PRED_DIR}/*_predictions.csv")

for PRED_PATH in prediction_files:
    model_name = os.path.basename(PRED_PATH).replace('_predictions.csv', '')
    print(f"\n=========================================")
    print(f"Evaluating Backtest for {model_name.upper()}")
    print(f"=========================================")
    
    pred = pd.read_csv(PRED_PATH)
    
    # Align data with the gap used in testing
    aligned_data = data.iloc[-len(pred):].reset_index(drop=True)
    df = pd.concat([aligned_data, pred], axis=1)

    # Future return is tracked tick-by-tick so we accurately measure compounding equity
    df["eval_return"] = df["close"].pct_change().shift(-1)

    print("\nPrediction probability stats:")
    print(df["prob_up"].describe())

    thresholds = np.arange(0.51, 0.65, 0.005)
    fee = 0.0004

    print(f"\nTesting SOTA optimized thresholds for {model_name}...\n")

    best_return = -np.inf
    best_threshold = None
    best_curve = None

    for t in thresholds:
        temp = df.copy()
        temp["position"] = 0
        # SOTA Threshold signal filter
        temp.loc[temp.prob_up > t, "position"] = 1
        temp.loc[temp.prob_up < (1 - t), "position"] = -1
        
        # Avoid lookahead bias
        temp["position"] = temp["position"].shift(1)
        
        temp["strategy_return"] = temp["position"] * temp["eval_return"]
        
        trades = temp["position"].diff().abs()
        temp["strategy_return"] -= trades * fee
        
        temp["equity_curve"] = (1 + temp["strategy_return"].fillna(0)).cumprod()
        total_return = temp["equity_curve"].iloc[-1] - 1
        
        if temp["strategy_return"].std() != 0:
            sharpe = (temp["strategy_return"].mean() / temp["strategy_return"].std()) * np.sqrt(288 * 365)
        else:
            sharpe = 0
            
        peak = temp["equity_curve"].cummax()
        max_dd = ((temp["equity_curve"] - peak) / peak).min()
        trade_count = trades.sum()
        
        if trade_count > 10 and total_return > best_return:
            best_return = total_return
            best_threshold = t
            best_curve = temp
            
        print(f"Threshold: {t:.3f} | Trades: {int(trade_count)} | Ret: {total_return:.4f} | Sharpe: {sharpe:.2f}")

    print(f"\n---> {model_name} Best Valid Threshold: {best_threshold:.3f}")
    print(f"---> {model_name} Best Valid Return: {best_return:.4f}")

    if best_curve is not None:
        best_curve.to_csv(f"{RESULT_DIR}/{model_name}_best_strategy_equity.csv", index=False)
        plt.figure(figsize=(10,5))
        plt.plot(best_curve["equity_curve"], color='orange' if model_name == 'lightgbm' else 'blue' if model_name == 'random_forest' else 'green')
        plt.title(f"{model_name.upper()} SOTA Strategy Equity Curve")
        plt.xlabel("Time (5m bins)")
        plt.ylabel("Cumulative Equity")
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{model_name}_equity_curve.png")
        print(f"Saved best plot for {model_name} to {RESULT_DIR}/{model_name}_equity_curve.png")