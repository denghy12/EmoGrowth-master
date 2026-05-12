#!/usr/bin/env python

import argparse
import html
import json
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "ap",
    "f1",
    "precision",
    "recall",
    "support",
    "predicted_positive",
    "tp",
    "fp",
    "fn",
    "tn",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert *_per_class_metrics.csv files to wide class-by-task tables."
        )
    )
    parser.add_argument(
        "per_class_csv",
        nargs="*",
        help=(
            "Input *_per_class_metrics.csv files. If omitted, files are discovered "
            "under --detail-root."
        ),
    )
    parser.add_argument("--detail-root", default="./result_detail")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. When omitted, each table is written next "
            "to its input CSV."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for a single input file.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated per-class metric columns to spread across tasks.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write an HTML version with sticky headers for easier browsing.",
    )
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def discover_inputs(args):
    if args.per_class_csv:
        return [Path(path) for path in args.per_class_csv]
    return sorted(Path(args.detail_root).rglob("*_per_class_metrics.csv"))


def build_per_class_task_table(per_class_df, metrics):
    if per_class_df.empty:
        return pd.DataFrame(columns=["class_id", "class_name", "class_task"])

    metrics = [metric for metric in metrics if metric in per_class_df.columns]
    if "trained_task" not in per_class_df.columns or not metrics:
        return per_class_df.copy()

    table_df = per_class_df.copy()
    table_df["trained_task"] = pd.to_numeric(
        table_df["trained_task"], errors="coerce"
    )
    table_df = table_df.dropna(subset=["trained_task"])
    table_df["trained_task"] = table_df["trained_task"].astype(int)

    key_col = "class_id" if "class_id" in table_df.columns else "class_name"
    index_cols = [
        column
        for column in ["class_id", "class_name", "class_task"]
        if column in table_df.columns
    ]
    base = table_df[index_cols].drop_duplicates(subset=[key_col]).copy()
    if "class_id" in base.columns:
        base = base.sort_values("class_id")
    elif "class_name" in base.columns:
        base = base.sort_values("class_name")
    base = base.reset_index(drop=True)

    for task in sorted(table_df["trained_task"].unique()):
        task_df = table_df[table_df["trained_task"] == task].drop_duplicates(
            subset=[key_col], keep="last"
        )
        task_df = task_df.set_index(key_col)
        for metric in metrics:
            base[f"task{task}_{metric}"] = base[key_col].map(task_df[metric])

    return base


def output_path(input_path, args, inputs, extension):
    if args.output and len(inputs) == 1 and extension == ".csv":
        return Path(args.output)

    filename = input_path.name.replace(
        "_per_class_metrics.csv", f"_per_class_task_table{extension}"
    )
    if args.output_dir:
        detail_root = Path(args.detail_root).resolve()
        try:
            relative_parent = input_path.resolve().parent.relative_to(detail_root)
        except ValueError:
            relative_parent = Path()
        return Path(args.output_dir) / relative_parent / filename

    return input_path.with_name(filename)


def task_metric_column_sort_key(column):
    parts = column.split("_", 1)
    task_part = parts[0]
    if task_part.startswith("task") and task_part[4:].isdigit():
        return int(task_part[4:])
    return 10**9


def add_count_column(table):
    display_table = table.copy()
    if "count" in display_table.columns:
        return display_table
    if "次数" in display_table.columns:
        return display_table.rename(columns={"次数": "count"})

    support_columns = sorted(
        [
            column
            for column in display_table.columns
            if column.startswith("task") and column.endswith("_support")
        ],
        key=task_metric_column_sort_key,
    )
    if support_columns:
        count_values = display_table[support_columns].bfill(axis=1).iloc[:, 0]
    else:
        count_values = None

    insert_at = min(3, len(display_table.columns))
    display_table.insert(insert_at, "count", count_values)
    return display_table


def write_html(table, path, title):
    html_title = html.escape(title)
    display_table = add_count_column(table)
    table_for_json = display_table.astype(object).where(pd.notna(display_table), None)
    columns_json = json.dumps(list(table_for_json.columns), ensure_ascii=False)
    rows_json = json.dumps(
        table_for_json.to_dict(orient="records"), ensure_ascii=False
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .toolbar {{
      align-items: center;
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .segmented-control {{
      display: inline-flex;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
    }}
    .segmented-control button {{
      border: 0;
      border-right: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      cursor: pointer;
      font-size: 13px;
      padding: 7px 12px;
    }}
    .segmented-control button:last-child {{
      border-right: 0;
    }}
    .segmented-control button.active {{
      background: #0f172a;
      color: #ffffff;
    }}
    .layout-hint {{
      color: #64748b;
      font-size: 13px;
    }}
    .table-wrap {{
      max-height: calc(100vh - 130px);
      overflow: auto;
      border: 1px solid #d1d5db;
    }}
    table.metric-table {{
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric-table th,
    .metric-table td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: right;
    }}
    .metric-table th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      z-index: 2;
    }}
    .metric-table th:nth-child(1),
    .metric-table td:nth-child(1) {{
      position: sticky;
      left: 0;
      min-width: 64px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(2),
    .metric-table td:nth-child(2) {{
      position: sticky;
      left: 64px;
      min-width: 150px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(3),
    .metric-table td:nth-child(3) {{
      position: sticky;
      left: 214px;
      min-width: 80px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(4),
    .metric-table td:nth-child(4) {{
      position: sticky;
      left: 294px;
      min-width: 80px;
      text-align: right;
      background: #ffffff;
      z-index: 1;
      box-shadow: 1px 0 0 #e5e7eb;
    }}
    .metric-table th:nth-child(-n+4) {{
      background: #f3f4f6;
      z-index: 3;
    }}
  </style>
</head>
<body>
  <h1>{html_title}</h1>
  <div class="toolbar">
    <div class="segmented-control" role="group" aria-label="column layout">
      <button type="button" class="active" data-layout="task">按 task 分组</button>
      <button type="button" data-layout="metric">按指标分组</button>
    </div>
    <span class="layout-hint" id="layoutHint"></span>
  </div>
  <div class="table-wrap">
    <table class="metric-table" id="metricTable"></table>
  </div>
  <script>
    const tableColumns = {columns_json};
    const tableRows = {rows_json};
    const metricPreference = {json.dumps(DEFAULT_METRICS, ensure_ascii=False)};
    const frozenColumnCount = 4;
    const indexColumns = tableColumns.slice(0, frozenColumnCount);
    const metricColumns = tableColumns.slice(frozenColumnCount);

    function parseMetricColumn(column) {{
      const match = /^task(\\d+)_(.+)$/.exec(column);
      if (!match) {{
        return null;
      }}
      return {{
        column,
        task: Number(match[1]),
        metric: match[2],
      }};
    }}

    const parsedColumns = metricColumns
      .map(parseMetricColumn)
      .filter(Boolean);
    const tasks = [...new Set(parsedColumns.map((item) => item.task))]
      .sort((a, b) => a - b);
    const discoveredMetrics = [...new Set(parsedColumns.map((item) => item.metric))];
    const metrics = [
      ...metricPreference.filter((metric) => discoveredMetrics.includes(metric)),
      ...discoveredMetrics.filter((metric) => !metricPreference.includes(metric)),
    ];
    const knownMetricColumns = new Set(
      parsedColumns.map((item) => item.column)
    );
    const extraColumns = metricColumns.filter(
      (column) => !knownMetricColumns.has(column)
    );

    function taskFirstColumns() {{
      return [
        ...indexColumns,
        ...tasks.flatMap((task) =>
          metrics
            .map((metric) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function metricFirstColumns() {{
      return [
        ...indexColumns,
        ...metrics.flatMap((metric) =>
          tasks
            .map((task) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "NaN";
      }}
      if (typeof value === "number") {{
        return Number.isInteger(value) ? String(value) : value.toFixed(6);
      }}
      return String(value);
    }}

    function renderTable(columns) {{
      const table = document.getElementById("metricTable");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      columns.forEach((column) => {{
        const th = document.createElement("th");
        th.textContent = column;
        headerRow.appendChild(th);
      }});
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      tableRows.forEach((row) => {{
        const tr = document.createElement("tr");
        columns.forEach((column) => {{
          const td = document.createElement("td");
          td.textContent = formatValue(row[column]);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      table.replaceChildren(thead, tbody);
    }}

    function setLayout(layout) {{
      const isMetricLayout = layout === "metric";
      renderTable(isMetricLayout ? metricFirstColumns() : taskFirstColumns());
      document.querySelectorAll("[data-layout]").forEach((button) => {{
        button.classList.toggle("active", button.dataset.layout === layout);
      }});
      document.getElementById("layoutHint").textContent = isMetricLayout
        ? "当前列顺序：task0-7 的 ap，然后 task0-7 的 f1，依次类推。"
        : "当前列顺序：task0 的所有指标，然后 task1 的所有指标，依次类推。";
    }}

    document.querySelectorAll("[data-layout]").forEach((button) => {{
      button.addEventListener("click", () => setLayout(button.dataset.layout));
    }});
    setLayout("task");
  </script>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def export_table(input_path, args, inputs, metrics):
    per_class_df = pd.read_csv(input_path)
    table = build_per_class_task_table(per_class_df, metrics)

    csv_path = output_path(input_path, args, inputs, ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    if args.html:
        html_path = output_path(input_path, args, inputs, ".html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        write_html(table, html_path, input_path.stem)
        print(f"Wrote {html_path}")


def main():
    args = parse_args()
    inputs = discover_inputs(args)
    if not inputs:
        raise FileNotFoundError(
            f"No *_per_class_metrics.csv files found under {args.detail_root}"
        )
    if args.output and len(inputs) != 1:
        raise ValueError("--output can only be used with exactly one input file.")

    metrics = split_csv(args.metrics)
    for input_path in inputs:
        export_table(input_path, args, inputs, metrics)


if __name__ == "__main__":
    main()
