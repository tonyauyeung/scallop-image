import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
