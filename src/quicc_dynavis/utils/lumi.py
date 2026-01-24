# src/quicc_dynavis/utils/lumi.py

from pathlib import Path
import re
from typing import Iterable, List, Union

_RUN_RE = re.compile(r"^run\d+$")  # only run0, run1, ...

def discover_valid_runs(
    case_dir: Union[str, Path],
    exclude_keywords: Iterable[str] = ("abort",),
    allow_suffix_runs: bool = False,
) -> List[str]:
    """
    Discover run directories under <case_dir>/runs.

    Default behavior:
      - keep only run<integer> (run0, run1, ...)
      - exclude directories containing keywords like 'abort'
    """

    case_dir = Path(case_dir)
    runs_dir = case_dir / "runs"
    if not runs_dir.exists():
        raise RuntimeError(f"No runs/ directory in: {case_dir}")

    exclude_keywords = tuple(exclude_keywords)

    run_dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name

        if any(k in name for k in exclude_keywords):
            continue

        if allow_suffix_runs:
            if not name.startswith("run"):
                continue
        else:
            if not _RUN_RE.match(name):
                continue

        run_dirs.append(str(d))

    # numeric sort by run index
    def _run_index(p):
        m = re.search(r"run(\d+)", Path(p).name)
        return int(m.group(1)) if m else 10**9

    return sorted(run_dirs, key=_run_index)

