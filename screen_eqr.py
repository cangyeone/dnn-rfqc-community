"""After pip install, run: python screen_eqr.py /path/to/eqr [options]."""
import sys
from rfqc_bench.cli import main

if __name__ == '__main__':
    main(['screen-eqr', *sys.argv[1:]])
