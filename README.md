# Retell, Reward, Repeat using d-RLAIF

This repository contains the core implementation of the Retell, Reward, Repeat using d-RLAIF framework, specifically focused on **Narrativity Alignment** using Todorov's narrative stages.

## Core Contribution: Narrativity Alignment
The primary focus of this work is aligning large language models to complex structural narrative rubrics. 

- **`train_narrativity.py`**: The main training script. it implements dRLAIF focusing on Todorov narrative stages (Equilibrium, Disruption, Recognition, Attempt, New Equilibrium). It includes a specialized, progressive length penalty to ensure structural density without verbosity.
- **`prompt_builder.py`**: Contains the core logic for constructing evaluation prompts and the formal Todorov-inspired scoring rubrics (1-5 scale for narrativity).

## Secondary Implementation: Overall Alignment
As a baseline and broader application, we also include:
- **`train_overall.py`**: A general alignment script focusing on broader narrative quality (Conceivability, Coherence, and Structure) on a 1-3 scale.

## Utilities
- **`visualiser.py`**: A real-time monitoring dashboard designed to track structural reward components, penalties, and narrative length distributions during training.
- **`requirements.txt`**: Python dependencies.

## Setup

1. **Environment**: Install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

2. **Data**: The repository includes the **TimeTravel** dataset in the `./TimeTravel` folder. The scripts default to using `./TimeTravel/train_unsupervised.json`.

3. **Models**: Distributed training is handled via `accelerate`. Ensure `LLMS_PATH` points to your model weights directory.

## Usage

### Training for Narrativity
```bash
accelerate launch train_narrativity.py
```

### Training for Overall
```bash
accelerate launch train_overall.py
```

### Monitoring
```bash
export LOG_FILE="./path/to/your/output/logs/reward_components.jsonl"
python visualiser.py
```
