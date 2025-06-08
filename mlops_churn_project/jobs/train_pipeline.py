import sys
from pathlib import Path

# Dynamically add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preparation import load_data, prepare_data
from src.evaluate_model import report
from src.train_model import train

df_raw = load_data()
df = prepare_data(df_raw)
model = train(df)
report({"status": "Model trained and logged to MLflow."})
