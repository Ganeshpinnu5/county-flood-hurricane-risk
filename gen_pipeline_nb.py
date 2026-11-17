"""Regenerates pipeline.ipynb from pipeline.py.

pipeline.py is the single source of truth: it runs as a plain script on CREST
and splits on its `# %%` markers into the Colab notebook. Editing the .ipynb
directly means the next run of the .py silently disagrees with it.
"""

import json
import subprocess
from pathlib import Path

HEADER = """# County flood / hurricane risk -- training pipeline

Generated from `pipeline.py` by `gen_pipeline_nb.py`; edit the .py and
regenerate rather than editing here.

**Colab setup.** Runtime > Change runtime type > T4 GPU. Expects
`MyDrive/crisk/artifacts/daily_panel.parquet`, and beside it in
`MyDrive/crisk/`:

    cyclone_features.py
    labels.py
    fetch_nhc_adeck.py

Those three are imported by cell 4. The panel already carries the 21 cyclone
columns baked in, so the modules are needed for the feature list and the dense
imputation rule rather than to refetch any advisory -- the a-deck parquet does
not have to be on Drive.

Cell 1 stamps `CRISK_GIT_COMMIT` with the commit this notebook was generated
from, so the run's manifest can claim self-contained even though Colab has no
checkout. Add one cell above it holding the environment flags, because cell 3
reads them as it runs and a flag set afterwards has no effect:

    %env CRISK_COLD=1

`CRISK_COLD=1` refuses to warm-start from whatever is left in `models/`, so the
run's numbers depend only on its own directory and the panel hash. Set it for
any run that is going to be reported. Add `%env CRISK_SMOKE=1` alongside it to
rehearse on a short window, 150 counties and 2 epochs; drop that line for the
full run.

The last cell seals the run into a single zip, copies it to Drive and hashes
both sides. Download that zip -- Drive's mount buffers small writes and has
already silently dropped one run's results."""

MARKER = "# %% ="


def cells(src):
    out = []
    for i, chunk in enumerate(src.split(MARKER)):
        if i:  # everything after a marker starts mid-banner-comment
            chunk = chunk.split("\n", 1)[1]
        body = chunk.strip("\n")
        if not body:
            continue
        if i == 1:
            # The module docstring sits above the first marker; it belongs with
            # the imports rather than in a cell of its own.
            out[-1] += "\n\n" + body
        else:
            out.append(body)
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return ""


def stamp():
    """The commit this notebook is a faithful copy of, or nothing.

    A stamp from a dirty tree would let a Colab run claim self-contained
    against code the commit does not actually hold, so refuse one.
    """
    commit = _git("rev-parse", "HEAD")
    if not commit:
        print("no git checkout -- notebook carries no commit stamp")
        return None
    if _git("status", "--porcelain"):
        print("tree is dirty -- notebook carries no commit stamp")
        return None
    return f"%env CRISK_GIT_COMMIT={commit}"


def main():
    here = Path(__file__).parent
    code = cells((here / "pipeline.py").read_text())
    s = stamp()
    if s:
        code.insert(0, s)
    nb = {
        "cells": [{"cell_type": "markdown", "metadata": {},
                   "source": HEADER.splitlines(keepends=True)}]
        + [{"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": c.splitlines(keepends=True)}
           for c in code],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = here / "pipeline.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}  ({len(code)} code cells)")


if __name__ == "__main__":
    main()
