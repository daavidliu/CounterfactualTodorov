import json
import pandas as pd
import time
import os
import sys
import numpy as np

# --- CONFIG ---
LOG_FILE = os.getenv("LOG_FILE", "./output/logs/reward_components.jsonl") 
REFRESH_RATE = 2    

# Batch size calculation for H200: 24 (Batch per GPU) * 16 (Generations) = 384
BATCH_SIZE = 384

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_data():
    data = []
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    # Calculate step number based on lines processed
    if not df.empty:
        df['step'] = df.index // BATCH_SIZE
    return df

def monitor():
    print(f"--- Monitoring: {LOG_FILE} ---")
    print(f"--- Assumed Batch Size (generations per step): {BATCH_SIZE} ---")
    print("Waiting for data stream...")
    
    while True:
        df = load_data()
        
        # Check for the ACTUAL column output by your training script ('reward_val')
        if df.empty or 'reward_val' not in df.columns:
            time.sleep(2)
            continue
            
        # Select columns that actually exist in your logs
        # We want to track: Final Reward, Raw Judge Score, and Lengths
        target_cols = ['reward_val', 'raw_score', 'len_new', 'len_orig', 'penalty']
        existing_cols = [c for c in target_cols if c in df.columns]
        
        # Group by step and calculate mean
        grouped = df.groupby('step')[existing_cols].mean()
        recent = grouped.tail(20)
        
        clear_screen()
        print(f"=== dRLAIF TRAINING DASHBOARD ===")
        print(f"Total Gens: {len(df)} | Current Step: {grouped.index.max()}")
        print("-" * 85)
        print(f"{'STEP':<6} | {'REWARD':<8} | {'SCORE':<8} | {'LEN_NEW':<8} | {'LEN_REF':<8} | {'PENALTY':<8}")
        print("-" * 85)
        
        for step, row in recent.iterrows():
            reward = row.get('reward_val', 0)
            raw = row.get('raw_score', 0)
            l_new = row.get('len_new', 0)
            l_orig = row.get('len_orig', 0)
            pen = row.get('penalty', 0)
            
            # visual indicator for reward
            bar = "*" * int((reward - 1) * 5) if reward > 1 else ""
            
            print(f"{step:<6} | {reward:.4f} {bar:<3} | {raw:.2f}     | {l_new:.1f}    | {l_orig:.1f}    | {pen:.2f}")
            
        print("-" * 85)
        print(f"Last Updated: {time.strftime('%H:%M:%S')}")
        
        time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nExiting monitor.")