"""Compatibility entry point; use the same protocol/config as train.py."""

import runpy
import sys
from pathlib import Path


if __name__ == '__main__':
    sys.argv = [str(Path(__file__).with_name('train.py')), '--dataset', 'corel5k', *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name='__main__')
