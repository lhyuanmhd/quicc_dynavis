# #!/usr/bin/env python3
# import argparse
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")

# import sys
# sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
# from quicc_dynavis.timeseries import plot_timeseries

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("case_dir", nargs="?", default=".", help="Case dir (default: current directory)")
#     args = p.parse_args()

#     case_dir = Path(args.case_dir).resolve()
#     fig_dir = case_dir / "figures"
#     fig_dir.mkdir(parents=True, exist_ok=True)

#     plot_timeseries(folderFile=str(case_dir), save_dir=str(fig_dir), show=False)
#     print(f"[OK] timeseries saved under {fig_dir}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import contextlib
import io as pyio   # 用别名，避免和你自己的 io 模块混淆

import matplotlib
matplotlib.use("Agg")

sys.path.append('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')

from quicc_dynavis.timeseries import plot_timeseries
from quicc_dynavis import io   # 假定 print_simulation_summary 在这里


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "case_dir",
        nargs="?",
        default=".",
        help="Case dir (default: current directory)",
    )
    args = p.parse_args()

    case_dir = Path(args.case_dir).resolve()

    # ---- figures ----
    fig_dir = case_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_timeseries(
        folderFile=str(case_dir),
        save_dir=str(fig_dir),
        show=False,
    )
    print(f"[OK] timeseries saved under {fig_dir}")

    # ---- diagnostics / run_summary ----
    diag_dir = case_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    param_file = case_dir / "runs/run0" / "parameters.cfg"
    summary_file = diag_dir / "run_summary.txt"

    buf = pyio.StringIO()
    with contextlib.redirect_stdout(buf):
        io.print_simulation_summary(str(param_file))

    summary_file.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[OK] simulation summary written to {summary_file}")


if __name__ == "__main__":
    main()

