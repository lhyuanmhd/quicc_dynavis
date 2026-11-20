"""Utilities for writing QuICC simulation diagnostic summaries.

Case identifiers in older CSV files were saved with limited decimal precision.
Matching those rounded values using a fixed 1e-6 relative tolerance can fail.
The matcher below respects the actual CSV precision (capped by the known
legacy output format), while new case identifiers are saved at high precision.
"""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


DYNAMO_SUMMARY_HEADER = [
    "q", "Ra", "Ek", "Pm", "Pr", "E0mag",
    "dEm_dt", "dyn", "fdip", "Lambda",
    "visDis", "ohmDis", "fohm", "Ld_u", "Ld_b",
    "T_perb", "Nu", "rev", "Ro", "flow_degree",
    "degree_over_pi", "local_Ro", "Rm", "relative_std_fdip",
    "bc_mag", "bc_temp", "bc_vel", "N", "M", "L", "dt",
    "time_start", "time_end", "time_span",
    "averaging_start", "averaging_end", "averaging_span",
]

# Old CSVs wrote case keys using these formats. Limit the inferred rounding
# uncertainty to what those formats could have introduced; an entry manually
# written as "0.6" must NOT match a different case at q=0.62.
_LEGACY_KEY_FORMATS = {
    "q": ".3f",
    "Ra": ".4e",
    "Ek": ".3e",
    "Pm": ".2e",
    "Pr": ".2e",
    "E0mag": ".2e",
}
_CASE_KEY_COLUMNS = tuple(_LEGACY_KEY_FORMATS)
_SPECTRAL_COLUMNS = ("flow_degree", "degree_over_pi", "local_Ro")


def _format_control_parameter(value):
    """Format a control parameter for new CSV rows, retaining precision."""
    if value is None or np.isinf(value):
        return "inf"
    return f"{float(value):.15e}"


def _format_case_value(value):
    """Keep enough digits to distinguish nearby q, Ra, Ek and E0mag runs."""
    if value is None:
        return "nan"
    return f"{float(value):.15e}"


def _format_finite(value, spec, default="nan"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return format(number, spec) if np.isfinite(number) else default


def _format_summary_row(
    q, Ra, Ek, E0mag, dEm_dt, dynamo, dipolarity, Elsasser,
    visDis, ohmDis, fohm, Ld_u, Ld_b, T_perb, nusselt, reversal,
    Rm, relative_std_fdip, bc_mag, bc_temp, bc_vel, N, M, L,
    dt, time_start, time_end, time_span,
    averaging_start, averaging_end, averaging_span,
    *, Pm=np.inf, Pr=np.inf, Ro=np.nan, flow_degree=np.nan,
    degree_over_pi=np.nan, local_Ro=np.nan,
):
    """Format one dynamo diagnostic row for CSV output."""
    return [
        _format_case_value(q),
        _format_case_value(Ra),
        _format_case_value(Ek),
        _format_control_parameter(Pm),
        _format_control_parameter(Pr),
        _format_case_value(E0mag),
        _format_finite(dEm_dt, ".3e"),
        int(dynamo),
        f"{dipolarity:.2f}",
        f"{Elsasser:.2e}",
        f"{visDis:.2e}",
        f"{ohmDis:.2e}",
        f"{fohm:.3f}",
        f"{Ld_u:.2e}",
        f"{Ld_b:.2e}",
        f"{T_perb:.3e}",
        _format_finite(nusselt, ".2f"),
        int(reversal) if np.isfinite(reversal) else "nan",
        _format_finite(Ro, ".3e"),
        _format_finite(flow_degree, ".6g"),
        _format_finite(degree_over_pi, ".6g"),
        _format_finite(local_Ro, ".6g"),
        _format_finite(Rm, ".2f"),
        _format_finite(relative_std_fdip, ".2f"),
        bc_mag, bc_temp, bc_vel,
        int(N), int(M), int(L),
        _format_finite(dt, ".3e"),
        f"{time_start:.6g}",
        f"{time_end:.6g}",
        f"{time_span:.6g}",
        f"{averaging_start:.6g}",
        f"{averaging_end:.6g}",
        f"{averaging_span:.6g}",
    ]


def _legacy_quantum(value, column):
    """Largest possible rounding unit in the original summary output."""
    if column == "q":
        return 1e-3
    if value == 0:
        # Scientific notation at zero is not informative about the intended
        # exponent. Require near-exact agreement instead.
        return 0.0
    spec = _LEGACY_KEY_FORMATS[column]
    decimals = int(spec[1:-1])
    exponent = int(np.floor(np.log10(abs(value))))
    return 10.0 ** (exponent - decimals)


def _numeric_values_match(
    old_value, new_value, *, column=None, rtol=1e-12, atol=1e-15,
):
    """Match numeric values allowing for rounding in stored CSV text.

    For a legacy value '6.67e+01' (Pr), the CSV precision is 0.1, so
    66.667 matches. For a new high-precision value the allowed error is
    correspondingly smaller. The known legacy format limits the tolerance.
    """
    try:
        stored = float(old_value)
        actual = float(new_value)
    except (TypeError, ValueError):
        return False

    if np.isnan(stored) or np.isnan(actual):
        return False
    if np.isinf(stored) or np.isinf(actual):
        return stored == actual

    tolerance = max(atol, rtol * max(abs(stored), abs(actual)))
    if column in _LEGACY_KEY_FORMATS:
        try:
            decimal_value = Decimal(str(old_value).strip())
            stored_quantum = float(Decimal(1).scaleb(decimal_value.as_tuple().exponent))
        except (InvalidOperation, OverflowError, ValueError):
            stored_quantum = 0.0
        # Respect precision of the actual stored string, never more lenient
        # than the legacy summary format for the given physical parameter.
        quantum = min(abs(stored_quantum), _legacy_quantum(stored, column))
        tolerance += 0.5 * quantum

    return abs(stored - actual) <= tolerance


def _case_matches_row(row, header, *, q, Ra, Ek, Pm, Pr, E0mag):
    """Use one consistent matching rule for initial writes and later updates."""
    wanted = dict(q=q, Ra=Ra, Ek=Ek, Pm=Pm, Pr=Pr, E0mag=E0mag)
    return all(
        column in header
        and header.index(column) < len(row)
        and _numeric_values_match(
            row[header.index(column)], value, column=column,
        )
        for column, value in wanted.items()
    )


def _row_matches_case(row, q, Ra, Ek, E0mag, *, Pm=np.inf, Pr=np.inf):
    """Return whether a row using the current header belongs to this case."""
    return _case_matches_row(
        row, DYNAMO_SUMMARY_HEADER,
        q=q, Ra=Ra, Ek=Ek, Pm=Pm, Pr=Pr, E0mag=E0mag,
    )


def _summary_sort_key(row):
    """Return a numeric sorting key, placing invalid rows last."""
    try:
        return (0, *(float(row[i]) for i in range(6)))
    except (IndexError, TypeError, ValueError):
        return (1, *(float("inf") for _ in range(6)))


def _convert_summary_row(old_header, old_row):
    """Convert an older summary row to the current CSV format."""
    old_values = {
        column: old_row[index]
        for index, column in enumerate(old_header)
        if index < len(old_row)
    }
    if "Ld_u" not in old_values and "L_u" in old_values:
        old_values["Ld_u"] = old_values["L_u"]
    if "Ld_b" not in old_values and "L_b" in old_values:
        old_values["Ld_b"] = old_values["L_b"]
    defaults = {
        "Pm": "inf", "Pr": "inf", "Ro": "nan",
        "flow_degree": "nan", "degree_over_pi": "nan", "local_Ro": "nan",
    }
    return [old_values.get(col, defaults.get(col, "nan"))
            for col in DYNAMO_SUMMARY_HEADER]


def _read_summary_rows(csv_path):
    """Read and normalize an existing summary to the current header."""
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    header = rows[0]
    if header == DYNAMO_SUMMARY_HEADER:
        result = [list(row) for row in rows[1:] if row]
    elif header[:3] == ["q", "Ra", "Ek"]:
        result = [_convert_summary_row(header, row) for row in rows[1:] if row]
    else:
        raise ValueError(f"Unrecognized CSV header in {csv_path}: {header}")
    for row in result:
        if len(row) < len(DYNAMO_SUMMARY_HEADER):
            row.extend(["nan"] * (len(DYNAMO_SUMMARY_HEADER) - len(row)))
    return result


def _write_rows(csv_path, header, data_rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows)


def write_dynamo_summary_csv(
    csv_path, q, Ra, Ek, E0mag, dEm_dt, dynamo, dipolarity,
    Elsasser, visDis, ohmDis, fohm, Ld_u, Ld_b, T_perb,
    nusselt, reversal, Rm, relative_std_fdip, bc_mag, bc_temp,
    bc_vel, N, M, L, dt, time_start, time_end, time_span,
    averaging_start, averaging_end, averaging_span,
    *, Pm=np.inf, Pr=np.inf, Ro=np.nan, flow_degree=np.nan,
    degree_over_pi=np.nan, local_Ro=np.nan,
):
    """Add or update a simulation entry in a dynamo summary CSV file."""
    csv_path = Path(csv_path)
    new_row = _format_summary_row(
        q=q, Ra=Ra, Ek=Ek, E0mag=E0mag, dEm_dt=dEm_dt,
        dynamo=dynamo, dipolarity=dipolarity, Elsasser=Elsasser,
        visDis=visDis, ohmDis=ohmDis, fohm=fohm, Ld_u=Ld_u,
        Ld_b=Ld_b, T_perb=T_perb, nusselt=nusselt,
        reversal=reversal, Rm=Rm,
        relative_std_fdip=relative_std_fdip,
        bc_mag=bc_mag, bc_temp=bc_temp, bc_vel=bc_vel,
        N=N, M=M, L=L, dt=dt, time_start=time_start,
        time_end=time_end, time_span=time_span,
        averaging_start=averaging_start, averaging_end=averaging_end,
        averaging_span=averaging_span, Pm=Pm, Pr=Pr,
        Ro=Ro, flow_degree=flow_degree,
        degree_over_pi=degree_over_pi, local_Ro=local_Ro,
    )

    data_rows = _read_summary_rows(csv_path)
    indices = [
        i for i, row in enumerate(data_rows)
        if _row_matches_case(row, q, Ra, Ek, E0mag, Pm=Pm, Pr=Pr)
    ]
    if len(indices) > 1:
        raise ValueError(
            f"Ambiguous summary case match in {csv_path}: "
            f"{len(indices)} rows match q={q}, Ra={Ra}, Ek={Ek}, "
            f"Pm={Pm}, Pr={Pr}, E0mag={E0mag}. "
            "Some old keys may have been rounded too aggressively."
        )

    updated = bool(indices)
    if updated:
        index = indices[0]
        # Preserve spectral diagnostics from the previous spectra pass if
        # this write does not supply replacements.
        for column in _SPECTRAL_COLUMNS:
            idx = DYNAMO_SUMMARY_HEADER.index(column)
            if new_row[idx] == "nan":
                new_row[idx] = data_rows[index][idx]
        data_rows[index] = new_row
    else:
        data_rows.append(new_row)

    data_rows.sort(key=_summary_sort_key)
    _write_rows(csv_path, DYNAMO_SUMMARY_HEADER, data_rows)
    action = "Updated" if updated else "Added"
    print(
        f"[OK] {action}: q={q:.6g}, Ra={Ra:.2e}, Ek={Ek:.2e}, "
        f"Pm={_format_control_parameter(Pm)}, "
        f"Pr={_format_control_parameter(Pr)}"
    )
    print(f"[OK] CSV written to {csv_path} ({len(data_rows)} total entries)")


def update_dynamo_summary_spectra(
    csv_path, q, Ra, Ek, flow_degree, *, E0mag,
    Pm=np.inf, Pr=np.inf,
):
    """Update the spectral diagnostics for exactly one existing CSV case.

    The Rossby number stored in the CSV is used to calculate
    local_Ro = Ro * flow_degree / pi.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Summary CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Summary CSV is empty: {csv_path}")

    header = rows[0]
    data_rows = [row for row in rows[1:] if row]
    required_columns = list(_CASE_KEY_COLUMNS) + ["Ro"]
    missing = [col for col in required_columns if col not in header]
    if missing:
        raise ValueError("Summary CSV is missing required columns: "
                         + ", ".join(missing))

    for row in data_rows:
        row.extend(["nan"] * max(0, len(header) - len(row)))
    for column in _SPECTRAL_COLUMNS:
        if column not in header:
            header.append(column)
            for row in data_rows:
                row.append("nan")

    indices = [
        i for i, row in enumerate(data_rows)
        if _case_matches_row(
            row, header, q=q, Ra=Ra, Ek=Ek, Pm=Pm, Pr=Pr,
            E0mag=E0mag,
        )
    ]
    if not indices:
        raise ValueError(
            "Could not find matching case in summary CSV: "
            f"q={q}, Ra={Ra}, Ek={Ek}, Pm={Pm}, Pr={Pr}, "
            f"E0mag={E0mag}. Ensure this case has first been "
            "written by write_dynamo_summary_csv."
        )
    if len(indices) > 1:
        raise ValueError(
            f"Ambiguous spectral update in {csv_path}: "
            f"{len(indices)} rows match q={q}, Ra={Ra}, Ek={Ek}, "
            f"Pm={Pm}, Pr={Pr}, E0mag={E0mag}. "
            "Please recover more precise case identifiers."
        )

    row = data_rows[indices[0]]
    try:
        rossby_number = float(row[header.index("Ro")])
    except (TypeError, ValueError):
        rossby_number = np.nan

    degree_over_pi = flow_degree / np.pi
    local_rossby = (rossby_number * degree_over_pi
                    if np.isfinite(rossby_number) else np.nan)
    row[header.index("flow_degree")] = f"{flow_degree:.6g}"
    row[header.index("degree_over_pi")] = f"{degree_over_pi:.6g}"
    row[header.index("local_Ro")] = (
        f"{local_rossby:.6g}" if np.isfinite(local_rossby) else "nan"
    )
    _write_rows(csv_path, header, data_rows)
    print(
        "[OK] Updated spectral diagnostics: "
        f"flow_degree={flow_degree:.4f}, "
        f"degree_over_pi={degree_over_pi:.4f}"
    )
