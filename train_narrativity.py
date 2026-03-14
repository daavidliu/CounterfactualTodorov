# --- Setup and Environment ---
import os
import re
import json
import torch
import time
import inspect
import sys
import matplotlib
matplotlib.use('Agg') # Server-safe backend
import matplotlib.pyplot as plt
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from accelerate import Accelerator

try:
    from prompt_builder import build_eval_prompt
except ImportError:
    raise ImportError("Could not find 'prompt_builder.py' in the current directory.")

CUR_DIR = os.getcwd()
ATTN_IMPL = "sdpa"  
TORCH_DTYPE = torch.bfloat16

# --- Paths and Models ---
LLMS_PATH = os.getenv("LLMS_PATH", "/path/to/models")
DATA_PATH = os.getenv("DATA_PATH", "./TimeTravel/train_unsupervised.json")

POLICY_NAME = "allenai/Olmo-3-7B-Instruct"
JUDGE_NAME = "Selene-1-Mini-Llama-3.1-8B"

POLICY_PATH = f"{LLMS_PATH}/{POLICY_NAME}"
JUDGE_PATH = f"{LLMS_PATH}/{JUDGE_NAME}"

TIMESTAMP = time.strftime("%H%M")
OUTPUT_DIR = os.path.join(CUR_DIR, f"{POLICY_NAME.replace('/', '-')}-narrativity-{TIMESTAMP}")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "reward_components.jsonl")
DEBUG_FILE = os.path.join(LOG_DIR, "judge_raw_outputs.txt") 

RUBRICS = ['narrativity']
os.makedirs(LOG_DIR, exist_ok=True) 

# --- Setup Distributed Training ---
accelerator = Accelerator()
current_device = accelerator.local_process_index 

if accelerator.is_main_process:
    print(f">>> Distributed Training Detected. Using {accelerator.num_processes} GPUs.")
    print(f">>> Output Directory: {OUTPUT_DIR}")

# --- Dataset Loading and Formatting ---
if accelerator.is_main_process:
    print(">>> Loading Dataset...")
    
raw_dataset = load_dataset('json', data_files=DATA_PATH, split="train")
DATASET_LOOKUP = {str(i): item for i, item in enumerate(raw_dataset)}

if accelerator.is_main_process:
    print(f">>> Loading Tokenizers...")
    
tokenizer = AutoTokenizer.from_pretrained(POLICY_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_PATH)
judge_tokenizer.pad_token = judge_tokenizer.eos_token
judge_tokenizer.padding_side = "left"

def format_prompt_execution(example, idx):      
    user_content_str = (
        f"[[ ## premise ## ]]\n{example['premise']}\n\n"
        f"[[ ## initial ## ]]\n{example['initial']}\n\n"
        f"[[ ## original_ending ## ]]\n{example['original_ending']}\n\n"
        f"[[ ## counterfactual ## ]]\n{example['counterfactual']}\n\n"
        "Respond with the corresponding output fields, starting with the field `[[ ## edited_ending ## ]]`, "
        "and then ending with the marker for `[[ ## completed ## ]]`."
    )
    
    messages = [{"role": "user", "content": user_content_str}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": f"{prompt_str}\n<|id:{idx}|>"}

train_dataset = raw_dataset.map(format_prompt_execution, with_indices=True)

# --- Model Initialization ---
if accelerator.is_main_process:
    print(f">>> Loading Models...")

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_PATH,
    torch_dtype=TORCH_DTYPE,
    attn_implementation=ATTN_IMPL,
    device_map={"": current_device} 
).eval()

model = AutoModelForCausalLM.from_pretrained(
    POLICY_PATH,
    torch_dtype=TORCH_DTYPE,
    attn_implementation=ATTN_IMPL,
    device_map={"": current_device} 
)
model.config.use_cache = False

peft_config = LoraConfig(
    r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM", lora_dropout=0.05, bias="none"
)

# --- Reward and Scoring Logic ---
def extract_score(text):
    max_score = 5 if 'narrativity' in RUBRICS else 3
    
    # Try strict format
    match = re.search(r"(?:\[RESULT\]|<score>|\*\*Result:\*\*)\s*\(?(\d+)\)?", text, re.IGNORECASE)
    if match: 
        val = int(match.group(1))
        return min(max(val, 1), max_score)
    
    # Try loose format
    match_loose = re.search(r"(?:Rating|Score|Logic|Plausibility|Narrativity):\s*(\d+)", text, re.IGNORECASE)
    if match_loose: 
        val = int(match_loose.group(1))
        return min(max(val, 1), max_score)

    # Dynamic Fallback
    numbers = re.findall(rf'\b([1-{max_score}])\b', text) 
    if numbers: 
        return int(numbers[-1])
    
    return None

def reward_function(prompts, completions, **kwargs):
    """
    Main reward function for GRPOTrainer. 
    Applies the judge rubric and Calculates the Todorov length penalty.
    """
    all_inputs = []
    meta_map = [] 
    
    # --- A. PREPARE PROMPTS FOR JUDGE ---
    for i, (prompt_text, completion_text) in enumerate(zip(prompts, completions)):
        id_match = re.search(r"<\|id:(\d+)\|>", prompt_text)
        if not id_match: continue
        
        row_id = id_match.group(1)
        original_data = DATASET_LOOKUP.get(row_id)
        if not original_data: continue

        for rubric in RUBRICS:
            user_content = build_eval_prompt(
                premise=original_data['premise'], 
                initial=original_data['initial'],
                original_ending=original_data['original_ending'],
                counterfactual=original_data['counterfactual'],
                new_ending=completion_text,
                model_name=JUDGE_NAME,
                evaluation_task=rubric
            )
            messages = [{"role": "user", "content": user_content}]
            txt = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_inputs.append(txt)
            meta_map.append((i, rubric, row_id)) 

    if not all_inputs: return [0.0] * len(prompts)

    # --- B. INFERENCE (JUDGE) ---
    indexed_prompts = sorted(enumerate(all_inputs), key=lambda x: len(x[1]))
    sorted_prompts = [x[1] for x in indexed_prompts]
    original_indices = [x[0] for x in indexed_prompts]
    
    JUDGE_BATCH_SIZE = 128 
    raw_scores = {} 
    
    debug_f = None
    if accelerator.is_main_process:
        debug_f = open(DEBUG_FILE, "a", encoding="utf-8")

    for start in range(0, len(sorted_prompts), JUDGE_BATCH_SIZE):
        batch = sorted_prompts[start:start+JUDGE_BATCH_SIZE]
        batch_orig_indices = original_indices[start:start+JUDGE_BATCH_SIZE]
        
        inputs = judge_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(current_device)
        
        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs, 
                max_new_tokens=16, 
                pad_token_id=judge_tokenizer.eos_token_id,
                do_sample=False
            )
            
        decoded = judge_tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for local_i, txt in enumerate(decoded):
            score = extract_score(txt)
            orig_idx = batch_orig_indices[local_i]

            if debug_f:
                debug_f.write(f"--- Item {orig_idx} | Score: {score} ---\n{txt.strip()}\n\n")
            
            if score is None: score = 1
            raw_scores[orig_idx] = score
            
    if debug_f:
        debug_f.close()

    # --- C. AGGREGATE & APPLY PENALTIES (UPDATED) ---
    final_rewards = [0.0] * len(prompts)
    log_entries = []

    for flat_idx, score in raw_scores.items():
        comp_idx, rubric, row_id = meta_map[flat_idx]
        original_data = DATASET_LOOKUP.get(row_id)
        
        current_reward = float(score)
        
        # --- HEAVY LENGTH PENALTY LOGIC ---
        original_len = len(original_data['original_ending'])
        new_len = len(completions[comp_idx])
        
        # 1. Define Thresholds
        # If new length > 2.0x original, penalize.
        UPPER_RATIO = 2.0  
        # If new length < 0.5x original, penalize (too short)
        LOWER_RATIO = 0.5  
        
        ratio = new_len / max(original_len, 1)
        penalty = 0.0

        if ratio > UPPER_RATIO:
            # PROGRESSIVE PENALTY:
            # (Ratio - Limit) * Factor
            # Example: If Ratio is 2.5 (Limit 1.5), Excess is 1.0. 
            # Penalty = 1.0 * 3.0 = 3.0 points deducted.
            excess = ratio - UPPER_RATIO
            penalty = excess * 3.0  # Increase this 3.0 to 5.0 for even harsher penalties
            
        elif ratio < LOWER_RATIO:
            # Flat penalty for being too short
            penalty = 1.5 
            
        # 2. Apply Penalty & Clamp
        # We clamp at 0.0 to prevent massive negative spikes destabilizing training
        final_reward_val = max(current_reward - penalty, 0.0)

        final_rewards[comp_idx] = final_reward_val

        if accelerator.is_main_process:
            log_entries.append({
                "timestamp": time.time(), 
                "raw_score": score, 
                "len_new": new_len,
                "len_orig": original_len,
                "penalty": penalty,
                "reward_val": final_reward_val
            })

    if accelerator.is_main_process:
        with open(LOG_FILE, "a") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
            
    return final_rewards

# --- Training Callbacks ---

class PlottingCallback(TrainerCallback):
    """
    Updates a plot of Loss and Reward every time logs are generated.
    Saves image to LOG_DIR/training_metrics.png
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.train_loss = []
        self.rewards = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only main process should write the file
        if not accelerator.is_main_process: 
            return

        # Extract values
        step = state.global_step
        loss = logs.get("loss", None)
        # TRL often logs 'reward' or 'reward_mean'
        reward = logs.get("reward", logs.get("reward_mean", None))

        # Update internal history
        # (We append if we find data; we carry forward previous value if missing to keep arrays aligned)
        if loss is not None or reward is not None:
            self.steps.append(step)
            
            if loss is not None:
                self.train_loss.append(loss)
            else:
                # Carrier forward last known loss or 0
                self.train_loss.append(self.train_loss[-1] if self.train_loss else 0)

            if reward is not None:
                self.rewards.append(reward)
            else:
                # Carry forward last known reward or 0
                self.rewards.append(self.rewards[-1] if self.rewards else 0)

        # Generate Plot
        try:
            plt.figure(figsize=(12, 5))

            # Subplot 1: Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.steps, self.train_loss, label='Loss', color='tab:red')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Step {step})')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Subplot 2: Reward
            plt.subplot(1, 2, 2)
            plt.plot(self.steps, self.rewards, label='Reward', color='tab:blue')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.title(f'Average Reward (Step {step})')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "training_metrics.png"))
            plt.close()
        except Exception as e:
            # Don't crash training if plotting fails
            print(f"Warning: Plotting failed: {e}")

class RewardEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=2.90, patience=10):
        self.threshold = threshold
        self.patience = patience
        self.counter = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 'reward' is the key TRL typically uses for the average reward
        # It might also be 'reward_mean' depending on version. 
        # We check both just to be safe.
        current_reward = logs.get("reward", logs.get("reward_mean", 0.0))
        
        if current_reward >= self.threshold:
            self.counter += 1
            print(f"\n[EarlyStopping] Reward {current_reward:.3f} >= {self.threshold} ({self.counter}/{self.patience})")
        else:
            self.counter = 0
            
        if self.counter >= self.patience:
            print(f"\n[EarlyStopping] Stopping training because reward has been > {self.threshold} for {self.patience} steps.")
            control.should_training_stop = True

# --- Main Training Loop ---
if __name__ == "__main__":
    if accelerator.is_main_process:
        if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
        if os.path.exists(DEBUG_FILE): os.remove(DEBUG_FILE) 


    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=24,
        num_generations=16,             # H200 Power Move: High stability
        gradient_accumulation_steps=2,  # Accumulate to simulate larger batch if needed
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=1,
        bf16=True, 
        logging_steps=10,
        save_steps=50,
        report_to="none",
        ddp_find_unused_parameters=False, 
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[
            RewardEarlyStoppingCallback(threshold=4.9, patience=20),
            PlottingCallback(log_dir=LOG_DIR)  # <--- ADDED HERE
        ]
    )

    if accelerator.is_main_process:
        print(f">>> Starting Training on {accelerator.num_processes} GPUs...")
        
    trainer.train()
    
    if accelerator.is_main_process:
        trainer.save_model(f"{OUTPUT_DIR}/final_adapter")