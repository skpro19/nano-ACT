"""Plot test 3's paired zero/prior reconstruction baselines."""
from pathlib import Path

from plot_reconstruction import main


if __name__ == '__main__':
    main(test=3, code=Path(__file__))
