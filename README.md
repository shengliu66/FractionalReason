# FractionalReason

A research framework for evaluating and improving large language model (LLM) reasoning through latent shifting techniques and advanced evaluation methods.

## Overview

FractionalReason is a **training-free and model-agnostic framework** that enables continuous control over reasoning intensity at inference time. Unlike existing test-time compute methods that apply reasoning uniformly across inputs, Fractional Reasoning recognizes that different problems require different levels of reasoning depth.

The framework operates by extracting **latent steering vectors** associated with deeper reasoning and reapplying them with tunable scaling factors, allowing models to tailor their reasoning process to the complexity of each input. This approach goes beyond the limitations of fixed instructional prompts and supports two key modes of test-time scaling:

1. **Breadth-based strategies**: Improving output quality through Best-of-N and majority voting
2. **Depth-based strategies**: Enhancing individual reasoning chain correctness through self-reflection

## Key Features

- **Continuous Reasoning Control**: Tunable scaling factors for dynamic reasoning intensity
- **Training-Free**: No model fine-tuning required - works with pre-trained models
- **Model-Agnostic**: Compatible with Qwen, Llama, R1Qwen, and other HuggingFace models
- **Dual Scaling Modes**: Support for both breadth-based and depth-based test-time scaling
- **Latent Steering**: Advanced latent space manipulation for reasoning enhancement
- **Comprehensive Evaluation**: Extensive testing on mathematical and scientific reasoning tasks
- **Parallel Execution**: SLURM-based scripts for efficient cluster computing

## Supported Models

- **Qwen2.5** 
- **Llama-3** 
- **R1Qwen** (reasoning-optimized variant)
- Any HuggingFace-compatible causal language model

## Supported Datasets

**Primary evaluation datasets (as tested in the paper):**
- **GSM8K**: Grade school math problems
- **MATH-500**: Mathematical reasoning (HuggingFaceH4/MATH-500)
- **GPQA**: Graduate-level science questions (Idavidrein/gpqa)

## Project Structure

```
FractionalReason/
├── exp/                          # Experimental scripts
│   ├── run_fr_with_matjority_vote.py    # Majority voting with Fractional Reasoning
│   ├── run_fr_with_rm.py                # Reward model scoring with latent steering
│   └── run_original_model.py            # Baseline model evaluation
├── utils/                        # Utility functions
│   ├── eval_utils.py            # Evaluation metrics and utilities
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── llm_layers.py            # Model layer modifications for latent steering
│   ├── score_outputs.py         # Output scoring and ranking
│   └── tools.py                 # General utility functions
├── tasks/                        # Task definitions and data loaders
│   ├── base.py                  # Base task class with steering vector computation
│   ├── loader.py                # Dataset loading utilities
│   └── demo.py                  # Demonstration task handler
├── models/                       # Model loading and configuration
│   └── huggingface.py           # HuggingFace model integration
├── scripts/                      # Batch execution scripts
│   ├── run_baselines.sh         # Run baseline evaluations
│   ├── run_ablation.sh          # Ablation studies
│   └── run_sns.sh               # SNS (Shift and Scale) experiments
├── records/                      # Experimental results
│   ├── rewarded_majority_vote/  # Results with reward model scoring
│   └── plain_majority_vote/     # Results with simple majority voting
├── anchor.py                     # Path configuration
└── common.py                     # Shared utilities and argument parsing
```

## Setup

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **CUDA-compatible GPU** (recommended for model inference)
- **Git LFS** (for large model files)
- At least **16GB GPU memory** for 7B models, **32GB+** for larger models

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd FractionalReason
```

2. **Create and activate a virtual environment:**
```bash
# Using conda (recommended)
conda create -n fractional-reason python=3.10
conda activate fractional-reason

# Or using venv
python -m venv fractional-reason
source fractional-reason/bin/activate  # On Windows: fractional-reason\Scripts\activate
```

3. **Install dependencies:**
```bash
# Install from requirements file
pip install -r requirements.txt

# Or install manually
pip install torch>=1.12.0 transformers>=4.21.0 datasets>=2.0.0 numpy>=1.21.0 tqdm>=4.64.0 bitsandbytes>=0.37.0
```

4. **Configure paths in `anchor.py`:**
```python
# Update the checkpoints_root to your HuggingFace cache directory
checkpoints_root = Path("/your/path/to/huggingface/cache")
```

2. **Model Download Issues:**
   ```bash
   # Set up Git LFS if not already installed
   git lfs install
   
   # Clear HuggingFace cache if corrupted
   rm -rf ~/.cache/huggingface/transformers
   ```

3. **Permission Errors with Models:**
   ```bash
   # Make sure you have access to gated models
   huggingface-cli login
   ```

4. **bitsandbytes Installation Issues:**
   ```bash
   # Install specific CUDA version
   pip install bitsandbytes-cuda111  # for CUDA 11.1
   pip install bitsandbytes-cuda118  # for CUDA 11.8
   ```

**Performance Tips:**
- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable gradient checkpointing for memory efficiency
- Use mixed precision training with `torch.autocast()`

## Usage

### Basic Evaluation

Run a basic evaluation with majority voting and Fractional Reasoning:

```bash
python exp/run_fr_with_matjority_vote.py \
    --model_type Qwen2.5 \
    --model_size 7b \
    --dataset gsm8k \
    --num_trials 5
```

### Fractional Reasoning with Reward Model

Use reward model scoring for sample selection:

```bash
python exp/run_fr_with_rm.py \
    --model_type Qwen2.5 \
    --model_size 7b \
    --dataset gsm8k \
    --num_trials 5
```

### Baseline Comparison

Run the original model without fractional reasoning for comparison:

```bash
python exp/run_original_model.py \
    --model_type Qwen2.5 \
    --model_size 7b \
    --dataset gsm8k
```

### Batch Execution

For cluster computing, use the provided shell scripts:

```bash
# Run all baseline evaluations
bash scripts/run_baselines.sh

# Run ablation studies
bash scripts/run_ablation.sh
```

## Key Parameters

### Fractional Reasoning Configuration
- `--alpha_mode`: Distribution for alpha scaling factors (`uniform`, `fixed`)
- `--alpha_a`, `--alpha_b`: Range for alpha values controlling reasoning intensity
- `--num_trials`: Number of trials per question for breadth-based scaling

### Model Configuration
- `--model_type`: Model family (Qwen2.5, llama-3, R1Qwen)
- `--model_size`: Model size (7b, 14b, 72b, etc.)
- `--in_8bit`: Enable 8-bit quantization for memory efficiency

### Evaluation Configuration
- `--dataset`: Target dataset for evaluation
- `--start_sample`, `--n_samples`: Sample range for evaluation
- `--num_return_sequences`: Number of generations per query

## Method Overview

### Fractional Reasoning Framework

Fractional Reasoning addresses the limitation of uniform reasoning application across diverse problem complexities. The method works through:

1. **Latent Steering Vector Extraction**: Analyzing hidden states to identify reasoning-associated directions in the model's latent space
2. **Tunable Scaling**: Applying extracted steering vectors with continuous scaling factors (alpha) to control reasoning intensity
3. **Adaptive Reasoning**: Allowing models to dynamically adjust reasoning depth based on problem complexity

### Test-Time Scaling Modes

**Breadth-Based Scaling** (Multiple Output Generation):
- **Best-of-N**: Generate N responses with varying reasoning intensities and select the highest-quality output
- **Majority Voting**: Generate multiple responses with different alpha values and aggregate through voting
- **Reward Model Scoring**: Use trained reward models to score and rank fractionally-reasoned outputs

**Depth-Based Scaling** (Single Chain Enhancement):
- **Self-Reflection**: Enhance individual reasoning chains by applying fractional reasoning to improve step-by-step correctness
- **Iterative Refinement**: Progressively apply stronger reasoning intensities to refine solutions

### Key Advantages

- **Problem-Adaptive**: Automatically adjusts reasoning intensity based on problem complexity
- **Continuous Control**: Fine-grained control over reasoning depth through alpha scaling
- **No Training Required**: Works with existing pre-trained models without additional training
- **Universal Compatibility**: Model-agnostic approach works across different architectures
- **Composable**: Can be combined with existing test-time compute strategies

## Results Structure

Results are saved in JSON format under the `records/` directory:
- Detailed per-question results with generated responses

## Contributing

1. Follow the existing code structure and naming conventions
2. Add appropriate documentation for new features
3. Test on multiple datasets before submitting changes
4. Update this README for significant feature additions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2025fractional,
  title={Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute},
  author={Liu, Sheng and Chen, Tianlang and Lu, Pan and Ye, Haotian and Chen, Yizheng and Xing, Lei and Zou, James},
  journal={arXiv preprint arXiv:2506.15882},
  year={2025}
}
}
```

## License

[License information]

## Contact

For questions or issues, please shengl@stanford.edu.
