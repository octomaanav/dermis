"""A small table engine for any dataset a user brings: CSV/TSV/JSON/JSONL from a URL or pasted text.

Pure Python (no pandas, which hosted Agentverse agents don't have). The agents' LLM drives it through
tools: describe the columns, filter, group and aggregate, pivot, correlate, and chart. The Dermis
recordings are available as the built-in dataset "dermis".
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict

import httpx

MAX_BYTES = 8_000_000
MAX_ROWS = 50_000
AGGS = ("count", "sum", "mean", "median", "min", "max", "nunique")
OPS = ("==", "!=", ">", ">=", "<", "<=", "contains", "in")


class DatasetError(ValueError):
    pass


class Table:
    def __init__(self, rows: list[dict], name: str):
        if not rows:
            raise DatasetError("the dataset has no rows")
        self.name = name
        self.columns = list(dict.fromkeys(k for r in rows[:200] for k in r))
        self.types = {c: _infer([r.get(c) for r in rows]) for c in self.columns}
        self.rows = [{c: _coerce(r.get(c), self.types[c]) for c in self.columns} for r in rows[:MAX_ROWS]]
        self.truncated = len(rows) > MAX_ROWS

    # -- views -----------------------------------------------------------------

    def schema(self) -> dict:
        cols = []
        for c in self.columns:
            vals = [r[c] for r in self.rows if r[c] is not None]
            info = {"column": c, "type": self.types[c], "non_null": len(vals), "unique": len(set(map(str, vals)))}
            if self.types[c] == "number" and vals:
                info.update(min=_r(min(vals)), max=_r(max(vals)), mean=_r(statistics.fmean(vals)),
                            median=_r(statistics.median(vals)))
            else:
                info["top"] = [f"{v} ({n})" for v, n in Counter(map(str, vals)).most_common(5)]
            cols.append(info)
        return {"dataset": self.name, "rows": len(self.rows), "truncated_at": MAX_ROWS if self.truncated else None,
                "columns": cols, "sample": self.rows[:5]}

    def filtered(self, filters: list | None) -> list[dict]:
        rows = self.rows
        for f in filters or []:
            col, op, val = (f.get("column"), f.get("op", "=="), f.get("value")) if isinstance(f, dict) else f
            col = self.col(col)
            if op not in OPS:
                raise DatasetError(f"unknown operator {op!r}; use one of {', '.join(OPS)}")
            rows = [r for r in rows if _match(r[col], op, val, self.types[col])]
        return rows

    def col(self, name: str | None) -> str:
        if name in self.columns:
            return name
        low = {c.lower(): c for c in self.columns}
        if name and name.lower() in low:
            return low[name.lower()]
        raise DatasetError(f"no column {name!r}; columns are {', '.join(self.columns)}")

    def query(self, filters=None, group_by=None, metrics=None, sort_by=None, descending=True,
              limit=20, columns=None) -> dict:
        rows = self.filtered(filters)
        group_by = [self.col(g) for g in (group_by or [])]
        metrics = metrics or ([{"column": None, "agg": "count"}] if group_by else [])
        if group_by:
            buckets: dict = defaultdict(list)
            for r in rows:
                buckets[tuple(r[g] for g in group_by)].append(r)
            out = []
            for key, bucket in buckets.items():
                row = dict(zip(group_by, key))
                for m in metrics:
                    agg = m.get("agg", "count")
                    col = self.col(m["column"]) if m.get("column") else None
                    row[_metric_name(col, agg)] = _agg([b[col] for b in bucket] if col else bucket, agg)
                out.append(row)
        elif metrics:
            row = {}
            for m in metrics:
                agg = m.get("agg", "count")
                col = self.col(m["column"]) if m.get("column") else None
                row[_metric_name(col, agg)] = _agg([r[col] for r in rows] if col else rows, agg)
            out = [row]
        else:
            keep = [self.col(c) for c in columns] if columns else self.columns
            out = [{c: r[c] for c in keep} for r in rows]
        if sort_by:
            key = sort_by if out and sort_by in out[0] else self.col(sort_by)
            out.sort(key=lambda r: (r.get(key) is None, r.get(key)), reverse=bool(descending))
            if descending:  # keep missing values last
                out.sort(key=lambda r: r.get(key) is None)
        total = len(out)
        return {"matched_rows": len(rows), "result_rows": total, "rows": out[: int(limit or 20)]}

    def pivot(self, index: str, columns: str, value: str | None, agg: str = "count", filters=None) -> dict:
        rows = self.filtered(filters)
        index, columns = self.col(index), self.col(columns)
        value = self.col(value) if value else None
        cells: dict = defaultdict(list)
        for r in rows:
            cells[(r[index], r[columns])].append(r[value] if value else r)
        idx = sorted({k[0] for k in cells}, key=_sortkey)
        cols = sorted({k[1] for k in cells}, key=_sortkey)
        table = [{index: i, **{str(c): _agg(cells[(i, c)], agg) if (i, c) in cells else None for c in cols}} for i in idx]
        return {"index": index, "columns": [str(c) for c in cols], "agg": agg, "value": value, "rows": table[:60]}

    def derive(self, name: str, expression: str) -> dict:
        """Add a computed column from a formula over other columns, e.g. "tip / total_bill * 100"."""
        if not re.fullmatch(r"[A-Za-z_][\w ]{0,40}", name or ""):
            raise DatasetError("column names must be letters, digits and underscores")
        fn = _compile(expression, self.columns)
        for r in self.rows:
            r[name] = fn(r)
        if name not in self.columns:
            self.columns.append(name)
        self.types[name] = _infer([r[name] for r in self.rows])
        vals = [r[name] for r in self.rows if r[name] is not None]
        return {"added": name, "expression": expression, "non_null": len(vals),
                "sample": vals[:5], **({"min": _r(min(vals)), "max": _r(max(vals))} if vals and self.types[name] == "number" else {})}

    def correlations(self, filters=None) -> dict:
        rows = self.filtered(filters)
        nums = [c for c in self.columns if self.types[c] == "number"]
        pairs = []
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                xs = [(r[a], r[b]) for r in rows if r[a] is not None and r[b] is not None]
                if len(xs) > 2:
                    try:
                        pairs.append({"a": a, "b": b, "r": _r(statistics.correlation([x for x, _ in xs], [y for _, y in xs])), "n": len(xs)})
                    except statistics.StatisticsError:
                        pass
        pairs.sort(key=lambda p: -abs(p["r"]))
        return {"numeric_columns": nums, "strongest": pairs[:15]}


# -- formulas: a strict whitelist parsed with ast (no eval of user text) ------------

_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "log": math.log, "sqrt": math.sqrt}


def _compile(expression: str, columns: list[str]):
    import ast
    import operator as op
    bins = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
            ast.Mod: op.mod, ast.Pow: op.pow}
    lookup = {c.lower().replace(" ", "_"): c for c in columns}
    src = re.sub(r"`([^`]+)`", lambda m: m.group(1).replace(" ", "_"), expression or "")
    try:
        tree = ast.parse(src, mode="eval").body
    except SyntaxError as e:
        raise DatasetError(f"can't read formula {expression!r}: {e.msg}") from e

    def build(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return lambda r: n.value
        if isinstance(n, ast.Name):
            col = lookup.get(n.id.lower())
            if col is None:
                raise DatasetError(f"unknown column {n.id!r} in formula; columns: {', '.join(columns)}")
            return lambda r: r[col] if isinstance(r[col], (int, float)) and not isinstance(r[col], bool) else None
        if isinstance(n, ast.BinOp) and type(n.op) in bins:
            a, b, f = build(n.left), build(n.right), bins[type(n.op)]
            return lambda r: (None if (x := a(r)) is None or (y := b(r)) is None else f(x, y))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            a = build(n.operand)
            sign = -1 if isinstance(n.op, ast.USub) else 1
            return lambda r: None if (x := a(r)) is None else sign * x
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            args, f = [build(x) for x in n.args], _FUNCS[n.func.id]
            return lambda r: None if None in (v := [g(r) for g in args]) else f(*v)
        raise DatasetError("formulas may use columns, numbers, + - * / // % **, and abs/round/min/max/log/sqrt")

    fn = build(tree)

    def safe(r):
        try:
            v = fn(r)
            return _r(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None
    return safe


# -- loading ---------------------------------------------------------------------

def load(source: str, name: str | None = None) -> Table:
    source = (source or "").strip()
    if not source:
        raise DatasetError("no dataset given: paste CSV/JSON or give a URL")
    if source.lower() == "dermis":
        from .viz import fleet_rows
        rows, _ = fleet_rows()
        return Table(rows, "dermis recordings (192 grasp episodes)")
    if re.match(r"https?://", source):
        return parse(fetch(source), name or source.rsplit("/", 1)[-1][:60] or "dataset")
    return parse(source, name or "pasted data")


def fetch(url: str) -> str:
    url = _raw_url(url)
    _guard(url)
    try:
        with httpx.stream("GET", url, timeout=25, follow_redirects=True,
                          headers={"User-Agent": "dermis-agents/0.3 (+https://github.com/Soham1010/dermis-agents)"}) as r:
            r.raise_for_status()
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise DatasetError(f"file is larger than {MAX_BYTES // 1_000_000} MB")
                chunks.append(chunk)
            return b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
    except httpx.HTTPError as e:
        raise DatasetError(f"couldn't download {url}: {e}") from e


def parse(text: str, name: str) -> Table:
    text = text.lstrip("﻿").strip()
    if text.startswith(("[", "{")):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), [data])
        rows = [_flatten(r) for r in data if isinstance(r, dict)]
        return Table(rows, name)
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not rows or len(rows[0]) < 2:
        raise DatasetError("couldn't read that as a table (CSV, TSV, JSON or JSON lines)")
    return Table(rows, name)


def looks_like_table(text: str) -> bool:
    """True when a chat message carries a pasted CSV/TSV/JSON table rather than a question."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    if lines[0].lstrip().startswith(("[", "{")):
        return True
    for d in (",", "\t", ";", "|"):
        widths = [len(r) for r in csv.reader(lines[:10], delimiter=d)]
        if widths[0] >= 2 and all(w == widths[0] for w in widths):
            return True
    return False


def find_url(text: str) -> str | None:
    m = re.search(r"https?://\S+?\.(?:csv|tsv|json|jsonl)(?:\?\S*)?(?=[\s)>\]]|$)", text, re.I)
    return m.group(0) if m else None


# -- helpers ---------------------------------------------------------------------

def _raw_url(url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else url


def _guard(url: str) -> None:
    host = re.sub(r"^https?://([^/:]+).*", r"\1", url).lower()
    if host in ("localhost", "0.0.0.0") or host.endswith(".local") or re.match(
            r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)", host):
        raise DatasetError("private network addresses aren't allowed")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)[:200]
        else:
            out[key] = v
    return out


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else v
    s = str(v).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        f = float(s)
        return None if math.isnan(f) else (int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f)
    except ValueError:
        return None


def _infer(values: list) -> str:
    present = [v for v in values if v not in (None, "", "NA", "N/A", "null", "None", "nan")]
    if not present:
        return "text"
    if all(isinstance(v, bool) or str(v).lower() in ("true", "false") for v in present):
        return "bool"
    nums = sum(_num(v) is not None for v in present)
    return "number" if nums / len(present) >= 0.9 else "text"


def _coerce(v, typ: str):
    if v in (None, "", "NA", "N/A", "null", "None", "nan"):
        return None
    if typ == "number":
        return _num(v)
    if typ == "bool":
        return v if isinstance(v, bool) else str(v).lower() == "true"
    return str(v)


def _match(cell, op: str, val, typ: str) -> bool:
    if op == "in":
        vals = val if isinstance(val, list) else [val]
        return str(cell).lower() in {str(x).lower() for x in vals}
    if op == "contains":
        return cell is not None and str(val).lower() in str(cell).lower()
    if typ == "number":
        cell, val = cell, _num(val)
        if cell is None or val is None:
            return op == "!=" and not (cell is None and val is None)
    elif typ == "bool":
        val = val if isinstance(val, bool) else str(val).lower() == "true"
    else:
        cell, val = (str(cell).lower() if cell is not None else None), str(val).lower()
    try:
        return {"==": cell == val, "!=": cell != val, ">": cell > val, ">=": cell >= val,
                "<": cell < val, "<=": cell <= val}[op]
    except TypeError:
        return False


def _agg(values: list, agg: str):
    if agg == "count":
        return len(values)
    vals = [v for v in values if v is not None]
    if agg == "nunique":
        return len(set(map(str, vals)))
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    fn = {"sum": sum, "mean": statistics.fmean, "median": statistics.median, "min": min, "max": max}.get(agg)
    if fn is None:
        raise DatasetError(f"unknown aggregation {agg!r}; use one of {', '.join(AGGS)}")
    return _r(fn(nums))


def _metric_name(col: str | None, agg: str) -> str:
    return "count" if agg == "count" and not col else f"{agg}_{col}"


def _r(x):
    return round(x, 4) if isinstance(x, float) else x


def _sortkey(v):
    return (v is None, str(type(v)), v if v is not None else 0)
