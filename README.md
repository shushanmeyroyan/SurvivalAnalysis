# SurvivalAnalysis

Repository for Homework 3: Survival Analysis.

## Contents


- main.py — full working Python script that loads data/telco.csv, preprocesses the data,
fits AFT models, selects the final model, computes CLV, and saves outputs.
- survival_notebook.ipynb — markdown with the codes mirroring main.py.
- report.md — 2 paragraph report.
- requirements.txt — Python packages required.
- .gitignore — common ignores.


## How to run


1. Put telco.csv in a data/ folder next to main.py.
2. Install requirements: pip install -r requirements.txt.
3. Run: python/python3 main.py.
4. Outputs saved: output_with_clv.csv, survival_curves.png.