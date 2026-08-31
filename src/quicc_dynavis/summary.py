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
    "Ld_u",  # velocity dissipation length scale
    "Ld_b",  # magnetic dissipation length scale
    "T_perb",
    "Nu",
    "rev",
    "Ro",              # global Rossby number
    "flow_degree",     # energy-weighted spherical-harmonic degree
    "degree_over_pi",  # characteristic length scale for local Rossby number
    "local_Ro",        # local Rossby number
    "Rm",
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
    Ld_u,
    Ld_b,
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
    Ro=np.nan,
    flow_degree=np.nan,
    degree_over_pi=np.nan,
    local_Ro=np.nan,
):
    """Format one dynamo diagnostic row for CSV output."""

    return [
        f"{q:.2f}",
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
        f"{Ld_u:.2e}",
        f"{Ld_b:.2e}",
        f"{T_perb:.3e}",
        (
            f"{nusselt:.2f}"
            if np.isfinite(nusselt)
            else "nan"
        ),
        (
            int(reversal)
            if np.isfinite(reversal)
            else "nan"
        ),
        (
            f"{Ro:.3e}"
            if np.isfinite(Ro)
            else "nan"
        ),
        (
            f"{flow_degree:.6g}"
            if np.isfinite(flow_degree)
            else "nan"
        ),
        (
            f"{degree_over_pi:.6g}"
            if np.isfinite(degree_over_pi)
            else "nan"
        ),
        (
            f"{local_Ro:.6g}"
            if np.isfinite(local_Ro)
            else "nan"
        ),
        (
            f"{Rm:.2f}"
            if np.isfinite(Rm)
            else "nan"
        ),
        (
            f"{relative_std_fdip:.2f}"
            if np.isfinite(relative_std_fdip)
            else "nan"
        ),
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

def _convert_summary_row(
    old_header,
    old_row,
):
    """Convert an older summary row to the current CSV format."""

    old_values = {
        column: old_row[index]
        for index, column in enumerate(old_header)
        if index < len(old_row)
    }

    # Handle old dissipation-length column names.
    if "Ld_u" not in old_values and "L_u" in old_values:
        old_values["Ld_u"] = old_values["L_u"]

    if "Ld_b" not in old_values and "L_b" in old_values:
        old_values["Ld_b"] = old_values["L_b"]

    default_values = {
        "Pm": "inf",
        "Pr": "inf",
        "Ro": "nan",
        "flow_degree": "nan",
        "degree_over_pi": "nan",
        "local_Ro": "nan",
    }

    return [
        old_values.get(
            column,
            default_values.get(column, "nan"),
        )
        for column in DYNAMO_SUMMARY_HEADER
    ]

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
    Ld_u,
    Ld_b,
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
    Ro=np.nan,
    flow_degree=np.nan,
    degree_over_pi=np.nan,
    local_Ro=np.nan,
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
        Ld_u=Ld_u,
        Ld_b=Ld_b,
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
        Ro=Ro,
        flow_degree=flow_degree,
        degree_over_pi=degree_over_pi,
        local_Ro=local_Ro,
    )

    # data_rows = []

    # if csv_path.is_file():
    #     with csv_path.open("r", newline="", encoding="utf-8") as handle:
    #         rows = list(csv.reader(handle))

    #     if rows:
    #         first_row = rows[0]

    #         if first_row == DYNAMO_SUMMARY_HEADER:
    #             data_rows = rows[1:]

    #         elif first_row[:3] == ["q", "Ra", "Ek"]:
    #             # Convert rows from the old CSV format:
    #             #
    #             # q, Ra, Ek, E0mag, ...
    #             #
    #             # to the new format:
    #             #
    #             # q, Ra, Ek, Pm, Pr, E0mag, ...
    #             old_data_rows = rows[1:]

    #             data_rows = [
    #                 row[:3] + ["inf", "inf"] + row[3:]
    #                 for row in old_data_rows
    #                 if row
    #             ]

    #         else:
    #             raise ValueError(
    #                 f"Unrecognized CSV header in {csv_path}: "
    #                 f"{first_row}"
    #             )
    
    data_rows = []

    if csv_path.is_file():
        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.reader(handle))

        if rows:
            first_row = rows[0]

            if first_row == DYNAMO_SUMMARY_HEADER:
                data_rows = rows[1:]

            elif first_row[:3] == ["q", "Ra", "Ek"]:
                data_rows = [
                    _convert_summary_row(
                        old_header=first_row,
                        old_row=row,
                    )
                    for row in rows[1:]
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


def update_dynamo_summary_spectra(
    csv_path,
    q,
    Ra,
    Ek,
    flow_degree,
    *,
    Pm=np.inf,
    Pr=np.inf,
):
    """
    Update spectral diagnostics for one case in an existing summary CSV.

    The local Rossby number is calculated using the Rossby number already
    stored in the CSV:

        local_Ro = Ro * flow_degree / pi
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Summary CSV does not exist: {csv_path}"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"Summary CSV is empty: {csv_path}")

    header = rows[0]
    data_rows = rows[1:]


    for row in data_rows:
        if len(row) < len(header):
            row.extend(
                ["nan"] * (len(header) - len(row))
            )

    required_columns = [
        "q",
        "Ra",
        "Ek",
        "Pm",
        "Pr",
        "Ro",
    ]

    missing = [
        column for column in required_columns
        if column not in header
    ]

    if missing:
        raise ValueError(
            "Summary CSV is missing required columns: "
            + ", ".join(missing)
        )

    new_columns = [
        "flow_degree",
        "degree_over_pi",
        "local_Ro",
    ]

    for column in new_columns:
        if column not in header:
            header.append(column)
            for row in data_rows:
                row.append("nan")

    q_index = header.index("q")
    ra_index = header.index("Ra")
    ek_index = header.index("Ek")
    pm_index = header.index("Pm")
    pr_index = header.index("Pr")
    ro_index = header.index("Ro")

    flow_index = header.index("flow_degree")
    degree_pi_index = header.index("degree_over_pi")
    local_ro_index = header.index("local_Ro")

    degree_over_pi = flow_degree / np.pi

    updated = False

    for row in data_rows:
        if not (
            _numeric_values_match(row[q_index], q)
            and _numeric_values_match(row[ra_index], Ra)
            and _numeric_values_match(row[ek_index], Ek)
            and _numeric_values_match(row[pm_index], Pm)
            and _numeric_values_match(row[pr_index], Pr)
        ):
            continue

        try:
            rossby_number = float(row[ro_index])
        except (TypeError, ValueError):
            rossby_number = np.nan

        if np.isfinite(rossby_number):
            local_rossby = rossby_number * degree_over_pi
        else:
            local_rossby = np.nan

        row[flow_index] = f"{flow_degree:.6g}"
        row[degree_pi_index] = f"{degree_over_pi:.6g}"
        row[local_ro_index] = (
            f"{local_rossby:.6g}"
            if np.isfinite(local_rossby)
            else "nan"
        )

        updated = True
        break

    if not updated:
        raise ValueError(
            "Could not find matching case in summary CSV: "
            f"q={q}, Ra={Ra}, Ek={Ek}, Pm={Pm}, Pr={Pr}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows)

    print(
        "[OK] Updated spectral diagnostics: "
        f"flow_degree={flow_degree:.4f}, "
        f"degree_over_pi={degree_over_pi:.4f}"
    )