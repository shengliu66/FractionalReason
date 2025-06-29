from pathlib import Path
import os

root = Path(__file__).parent
data_root = root.joinpath("data")
inference_root = root.joinpath("inference")

logger_root = root.joinpath("logger")
dump_root = root.joinpath("dump")

# modify to /your/folder/contains/huggingface/cache
checkpoints_root = Path("/gpfs/data/razavianlab/home/sl5924/llm/.cache/huggingface/transformers")

hf_datasets_root = root.joinpath("datasets")
