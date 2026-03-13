import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from stable_baselines3 import PPO
sys.path.append('src/models/rl')
from poly_env import PolymarketBTCEnv

DATA_PATH = "data/processed/btc_features_ml.csv"
PRED_PATH = "results/predictions/ensemble_predictions.csv"
MODEL_PATH = "models/rl/ppo_polymarket"
RESULT_DIR = "results/backtest"

os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading Test Environment for Autonomous RL Evaluation...")

# We calculate the split in the exact same manner we did for ML
df = pd.read_csv(DATA_PATH)
split = int(len(df) * 0.8)
gap = 6

# We constrain the environment TO ONLY process the blind test data section 
test_df = df.iloc[split+gap:].reset_index(drop=True)

# Temporarily save out just the test section so the Gym environment loads it purely
test_df_path = "data/processed/temp_rl_test.csv"
test_df.to_csv(test_df_path, index=False)

# Match predictions subset
pred = pd.read_csv(PRED_PATH)
RL_pred_path = "results/predictions/temp_rl_pred.csv"
pred.to_csv(RL_pred_path, index=False)

# Initialize Test Env
env = PolymarketBTCEnv(data_path=test_df_path, pred_path=RL_pred_path)

print(f"Loading Fully Trained PPO Brain from {MODEL_PATH}.zip...")
model = PPO.load(MODEL_PATH)

obs, info = env.reset()

balances = [env.balance]
actions_taken = []
rewards = []

print("\nDeploying Autonomous Agent over out-of-sample data...")

done = False
while not done:
    # Pass observation to trained policy net (deterministic=True removes randomness)
    action, _states = model.predict(obs, deterministic=True)
    
    # Take action in environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    balances.append(info['balance'])
    actions_taken.append(action)
    rewards.append(reward)
    
    done = terminated or truncated

# Clean up temporary test files
os.remove(test_df_path)
os.remove(RL_pred_path)

# Calculate Evaluation Metrics
balances = np.array(balances)
actions_taken = np.array(actions_taken)
rewards = np.array(rewards)

# Evaluate discrete win loss counts
win_bets = len(rewards[rewards > 0])
loss_bets = len(rewards[rewards < 0])
total_trades = win_bets + loss_bets

hold_actions = np.sum(actions_taken == 0)
buy_up = np.sum(actions_taken == 1)
buy_down = np.sum(actions_taken == 2)

final_equity = balances[-1]
profit = final_equity - env.initial_balance

print("\n==========================================")
print("RL POLYMARKET BACKTEST RESULTS")
print("==========================================")
print(f"Total Ticks Evaluated: {len(actions_taken)}")
print(f"Total Bets Placed:     {total_trades}")
print(f"Actions [Hold: {hold_actions} | UP: {buy_up} | DOWN: {buy_down}]")

if total_trades > 0:
    win_rate = (win_bets / total_trades) * 100
    print(f"Absolute Win Rate:     {win_rate:.2f}%")
else:
    print("Absolute Win Rate:     0.00% (No trades)")

print(f"Starting Bankroll:     ${env.initial_balance:.2f}")
print(f"Ending Bankroll:       ${final_equity:.2f}")
print(f"Net Profit:            ${profit:.2f}")

# Plotting
plt.figure(figsize=(12,6))
plt.plot(balances, color='purple', linewidth=2)
plt.title("RL Agent Polymarket Bankroll Equity (OOS)")
plt.xlabel("Time (5m bins)")
plt.ylabel("Bankroll ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = f"{RESULT_DIR}/ppo_polymarket_equity.png"
plt.savefig(chart_path)
print(f"\nSaved Equity Curve to {chart_path}")
