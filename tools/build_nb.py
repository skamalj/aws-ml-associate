"""Convert a jupytext 'percent' script (.py) into a notebook (.ipynb).

Usage:  python tools/build_nb.py src.py dest.ipynb
"""
import sys
import jupytext

nb = jupytext.read(sys.argv[1], fmt="py:percent")
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.pop("jupytext", None)
jupytext.write(nb, sys.argv[2], fmt="ipynb")
print("wrote", sys.argv[2], len(nb.cells), "cells")
