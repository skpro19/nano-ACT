"""Plot test 2's paired arbitrary and proprio-matched posterior swaps."""
from pathlib import Path

from plot_reconstruction import main


if __name__ == '__main__':
    main(test=2, code=Path(__file__))
