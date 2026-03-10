"""Parse tables from OCR text (Markdown format)."""

import re
from typing import Any


def _parse_table_rows(lines: list[str], keep_separator: bool = False) -> list[list[str]]:
    """Parse pipe-delimited rows into cells. Returns list of row cells."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        # Split by |, strip each cell
        raw = [c.strip() for c in line.split("|")]
        # Drop leading/trailing empty cells from pipe boundaries
        cells = [c for c in raw if c]
        if not cells:
            continue
        is_separator = all(re.match(r"^[-:]+$", c) for c in cells)
        if is_separator and not keep_separator:
            continue
        rows.append(cells)
    return rows


def _split_header_and_data(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Split parsed rows into header rows and data rows. Supports multi-row headers."""
    if not rows:
        return [], []
    # Find separator row (all cells are --- or :---)
    sep_idx = -1
    for idx, row in enumerate(rows):
        if row and all(re.match(r"^[-:]+$", c) for c in row):
            sep_idx = idx
            break
    if sep_idx >= 0:
        header_rows = rows[:sep_idx]
        data_rows = rows[sep_idx + 1 :]
    else:
        header_rows = [rows[0]] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
    return header_rows, data_rows


def _build_header_groups(headers: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Parse Parent::Sub headers into row1 (parent row with colspan) and row2 (child row).
    Returns (row1_cells, row2_cells). row1 has {text, colSpan?, rowSpan?}. row2 has {text} or None for covered.
    """
    row1: list[dict[str, Any]] = []
    row2: list[dict[str, Any]] = []
    i = 0
    while i < len(headers):
        h = headers[i]
        if "::" in h:
            parent, sub = h.split("::", 1)
            group = [sub]
            j = i + 1
            while j < len(headers) and headers[j].startswith(parent + "::"):
                group.append(headers[j].split("::", 1)[1])
                j += 1
            row1.append({"text": parent, "colSpan": len(group)})
            for child in group:
                row2.append({"text": child})
            i = j
        else:
            row1.append({"text": h, "rowSpan": 2})
            row2.append(None)
            i += 1
    return row1, row2


def _resolve_rowspans(rows: list[list[str]]) -> list[list[dict[str, Any] | str]]:
    """
    Replace ^ with value from above, then compute rowSpan for consecutive identical values.
    Returns rows of cells: {value, rowSpan} or "covered" for cells covered by rowSpan above.
    """
    ROWSPAN_MARKER = "^"
    # Resolve ^ to value from above
    resolved: list[list[str]] = []
    max_cols = max(len(r) for r in rows) if rows else 0
    for r, row in enumerate(rows):
        new_row = []
        for c in range(max_cols):
            cell = row[c] if c < len(row) else ""
            if cell.strip() == ROWSPAN_MARKER and r > 0 and c < len(resolved[r - 1]):
                new_row.append(resolved[r - 1][c])
            else:
                new_row.append(cell)
        resolved.append(new_row)

    # Compute rowSpan
    result: list[list[dict[str, Any] | str | None]] = [
        [None] * max_cols for _ in resolved
    ]
    for c in range(max_cols):
        r = 0
        while r < len(resolved):
            if r < len(result) and c < len(result[r]) and result[r][c] is not None:
                r += 1
                continue
            val = resolved[r][c] if c < len(resolved[r]) else ""
            span = 1
            rr = r + 1
            while rr < len(resolved) and c < len(resolved[rr]) and resolved[rr][c] == val:
                span += 1
                rr += 1
            result[r][c] = {"value": val, "rowSpan": span}
            for rr in range(r + 1, r + span):
                if rr < len(result) and c < len(result[rr]):
                    result[rr][c] = "covered"
            r += span

    return result


def extract_tables_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract Markdown-style tables from OCR text.
    Returns (text_without_tables, list of table dicts).
    Each table: {"headers": [...], "header_rows": [[...], ...], "rows": [[...], ...]}.
    For simple tables: headers = first row. For complex tables: header_rows may have multiple rows.
    """
    tables: list[dict[str, Any]] = []
    result_lines: list[str] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        # Look for table start: line with |
        if "|" in line and (line.strip().startswith("|") or "|" in line.strip()):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                table_lines.append(lines[j])
                j += 1
            i = j

            rows = _parse_table_rows(table_lines, keep_separator=True)
            if rows:
                header_rows, data_rows = _split_header_and_data(rows)
                headers = header_rows[0] if header_rows else []
                table = {
                    "headers": headers,
                    "header_rows": header_rows,
                    "rows": data_rows,
                }
                # Nested headers: Parent::Sub format
                if any("::" in h for h in headers):
                    row1, row2 = _build_header_groups(headers)
                    table["header_structure"] = {"row1": row1, "row2": row2}
                # Row-spans: ^ marker
                if data_rows and any(
                    c.strip() == "^" for row in data_rows for c in row
                ):
                    table["row_cells"] = _resolve_rowspans(data_rows)
                tables.append(table)
                result_lines.append(f"\n[Table {len(tables)}]\n")
            continue

        result_lines.append(line)
        i += 1

    text_clean = "\n".join(result_lines)
    return text_clean, tables


def parse_tables_in_page(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Parse tables from page text. Returns (cleaned_text, tables).
    Tables are list of {"headers": [...], "rows": [[...]]}.
    """
    return extract_tables_from_text(text)
