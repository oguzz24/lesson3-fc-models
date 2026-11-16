"""Run all required FC experiments sequentially with shared CLI settings."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for experiment in ('depth', 'width', 'regularization'):
        subprocess.run([sys.executable, str(root / 'homework' / f'homework_{experiment}_experiments.py'), *sys.argv[1:]], check=True)
