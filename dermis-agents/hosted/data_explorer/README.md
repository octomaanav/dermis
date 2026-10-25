# Data Analyst Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **analysis of any dataset you bring (CSV, TSV, JSON or JSON lines by URL or pasted into the chat) and of the Dermis recordings: describe, filter, group, pivot, computed columns, correlations, charts**.

Focus: correct numbers from the data, the table or chart that answers the question, the reproducible pandas expression.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: table, tables, pivot, crosstab, groupby, group by, describe, stats, statistics, breakdown, columns, rows, pandas, dataframe, spreadsheet, explore, csv, json, dataset, my data, analyze this, correlation

Example questions:
- Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
- Paste a CSV and ask a question about it
- Peak force by object and scenario
- Which columns correlate most?

