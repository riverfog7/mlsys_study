import os
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_HOME = SCRIPT_DIR / ".hf_home"
RANDOM_SEED = 42
DEVICE = torch.device(
    "mps" if torch.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.environ["HF_HOME"] = HF_HOME.absolute().as_posix()
