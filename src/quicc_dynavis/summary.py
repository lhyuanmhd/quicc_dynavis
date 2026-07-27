"""Utilities for writing QuICC simulation diagnostic summaries."""

import csv
from pathlib import Path

import numpy as np


DYNAMO_SUMMARY_HEADER = [
    "q",
    "Ra",
    "Ek",
    "Pm",
    "Pr",
    "E0mag",
    "dyn",
    "fdip",
    "Lambda",
    "visDis",
    "ohmDis",
    "fohm",
    "L_u",  # velocity dissipation length scale
    "L_b",  # magnetic dissipation length scale
    "T_perb",
    "Nu",
    "rev",
    "Rm",
    "Ro"
    "relative_std_fdip",
    "bc_mag",
    "bc_temp",
    "bc_vel",
    "N",
    "M",
    "L",
]


def _format_control_parameter(value):
    """Format a finite control parameter or infinity for CSV output."""
    if value is None or np.isinf(value):
        return "inf"

    return f"{value:.2e}"


def _format_summary_row(
    q,
    Ra,
    Ek,
    E0mag,
    dynamo,
    dipolarity,
    Elsasser,
    visDis,
    ohmDis,
    fohm,
    L_u,
    L_b,
    T_perb,
    nusselt,
    reversal,
    Rm,
    relative_std_fdip,
    bc_mag,
    bc_temp,
    bc_vel,
    N,
    M,
    L,
    *,
    Pm=np.inf,
    Pr=np.inf,
    Ro
):
    """Format one dynamo diagnostic row for CSV output."""
    return [
        f"{q:.6g}",
        f"{Ra:.2e}",
        f"{Ek:.2e}",
        _format_control_parameter(Pm),
        _format_control_parameter(Pr),
        f"{E0mag:.2e}",
        int(dynamo),
        f"{dipolarity:.2f}",
        f"{Elsasser:.2e}",
        f"{visDis:.2e}",
        f"{ohmDis:.2e}",
        f"{fohm:.3f}",
        f"{L_u:.2e}",
        f"{L_b:.2e}",
        f"{T_perb:.3e}",
        f"{nusselt:.2f}" if not np.isnan(nusselt) else "nan",
        int(reversal) if not np.isnan(reversal) else "nan",
        f"{Rm:.2f}",
        f"{relative_std_fdip:.2f}",
        bc_mag,
        bc_temp,
        bc_vel,
        int(N),
        int(M),
        int(L),
    ]


def _numeric_values_match(old_value, new_value):
    """Compare finite or infinite numeric values."""
    try:
        old_value = float(old_value)
        new_value = float(new_value)
    except (TypeError, ValueError):
        return False

    if np.isinf(old_value) or np.isinf(new_value):
        return old_value == new_value

    return np.isclose(
        old_value,
        new_value,
        rtol=1e-6,
        atol=1e-12,
    )


def _row_matches_case(
    row,
    q,
    Ra,
    Ek,
    *,
    Pm=np.inf,
    Pr=np.inf,
):
    """Return whether an existing CSV row corresponds to the same case."""
    if len(row) < 5:
        return False

    return (
        _numeric_values_match(row[0], q)
        and _numeric_values_match(row[1], Ra)
        and _numeric_values_match(row[2], Ek)
        and _numeric_values_match(row[3], Pm)
        and _numeric_values_match(row[4], Pr)
    )


def _summary_sort_key(row):
    """Return a numeric sorting key, placing invalid rows last."""
    try:
        return (
            0,
            float(row[0]),  # q
            float(row[1]),  # Ra
            float(row[2]),  # Ek
            float(row[3]),  # Pm
            float(row[4]),  # Pr
        )
    except (IndexError, TypeError, ValueError):
        return (
            1,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )


def write_dynamo_summary_csv(
    csv_path,
    q,
    Ra,
    Ek,
    E0mag,
    dynamo,
    dipolarity,
    Elsasser,
    visDis,
    ohmDis,
    fohm,
    L_u,
    L_b,
    T_perb,
    nusselt,
    reversal,
    Rm,
    relative_std_fdip,
    bc_mag,
    bc_temp,
    bc_vel,
    N,
    M,
    L,
    *,
    Pm=np.inf,
    Pr=np.inf,
    Ro,
):
    """Add or update one simulation entry in a dynamo summary CSV file."""
    csv_path = Path(csv_path)

    new_row = _format_summary_row(
        q=q,
        Ra=Ra,
        Ek=Ek,
        E0mag=E0mag,
        dynamo=dynamo,
        dipolarity=dipolarity,
        Elsasser=Elsasser,
        visDis=visDis,
        ohmDis=ohmDis,
        fohm=fohm,
        L_u=L_u,
        L_b=L_b,
        T_perb=T_perb,
        nusselt=nusselt,
        reversal=reversal,
        Rm=Rm,
        relative_std_fdip=relative_std_fdip,
        bc_mag=bc_mag,
        bc_temp=bc_temp,
        bc_vel=bc_vel,
        N=N,
        M=M,
        L=L,
        Pm=Pm,
        Pr=Pr,
        Ro=Ro
    )

    data_rows = []

    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        if rows:
            first_row = rows[0]

            if first_row == DYNAMO_SUMMARY_HEADER:
                data_rows = rows[1:]

            elif first_row[:3] == ["q", "Ra", "Ek"]:
                # Convert rows from the old CSV format:
                #
                # q, Ra, Ek, E0mag, ...
                #
                # to the new format:
                #
                # q, Ra, Ek, Pm, Pr, E0mag, ...
                old_data_rows = rows[1:]

                data_rows = [
                    row[:3] + ["inf", "inf"] + row[3:]
                    for row in old_data_rows
                    if row
                ]

            else:
                raise ValueError(
                    f"Unrecognized CSV header in {csv_path}: "
                    f"{first_row}"
                )

    updated = False

    for index, row in enumerate(data_rows):
        if _row_matches_case(
            row,
            q=q,
            Ra=Ra,
            Ek=Ek,
            Pm=Pm,
            Pr=Pr,
        ):
            data_rows[index] = new_row
            updated = True
            break

    if not updated:
        data_rows.append(new_row)

    data_rows.sort(key=_summary_sort_key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DYNAMO_SUMMARY_HEADER)
        writer.writerows(data_rows)

    action = "Updated" if updated else "Added"

    print(
        f"[OK] {action}: "
        f"q={q:.6g}, "
        f"Ra={Ra:.2e}, "
        f"Ek={Ek:.2e}, "
        f"Pm={_format_control_parameter(Pm)}, "
        f"Pr={_format_control_parameter(Pr)}"
    )
    print(
        f"[OK] CSV written to {csv_path} "
        f"({len(data_rows)} total entries)"
    )
