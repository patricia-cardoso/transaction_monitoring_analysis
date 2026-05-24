# Transaction Monitoring Analysis

A Python notebook simulating a simplified transaction monitoring workflow across a portfolio of fictional banks.

Built as a small project to demonstrate data processing, quality checking, and visualisation skills in Python, as part of a job application to the ECB. Given that code developed in a professional context is confidential, this project was created specifically to showcase relevant technical skills using synthetic data.

---

## Files

**`transaction_monitoring_analysis.ipynb`**: Jupyter notebook version. More interactive — run cells one by one and see outputs inline.

**`transaction_monitoring_analysis.py`**: Python script version. Runs end to end from the terminal, printing all outputs and saving charts as PNG files.

---

## Project Structure

The analysis is organised in five sections:

1. **Data Generation**: synthetic transaction data for 34 fictional banks over 2025, with realistic trading profiles and deliberately injected data quality issues
2. **Data Quality Checks**: five checks covering missing values, duplicates, negative values, weekend transactions, and a sanity check on calculated fields
3. **Data Compilation**: aggregation from transaction level to monthly level per bank, deriving key metrics including month-over-month change
4. **Data Visualisation**: four charts exploring trading activity, amount vs frequency, counterparty concentration, and monthly volatility
5. **Analysis**: statistical summary of key findings across the dataset

---

## Charts

**Chart 1: Line chart:** Monthly trading activity for the top N most active banks, showing both total amount traded and trade count in two panels.

**Chart 2: Scatter plot:** Trading amount vs frequency across all banks, identifying banks with unusual profiles such as high amount but low frequency or vice versa.

**Chart 3: Heatmap matrix:** Number of trades between bank pairs for the top N most active banks, revealing counterparty concentration and network structure.

**Chart 4: Heatmap:** Month-over-month change in trading volume per bank, showing which banks have the most volatile activity over time.

---

## How to Run

**Notebook version:**

1. Install dependencies:
pip install -r requirements.txt

2. Open the notebook:
jupyter notebook transaction_monitoring_analysis.ipynb

3. Run all cells from top to bottom.

**Script version:**

1. Install dependencies:
pip install -r requirements.txt

2. Run the script:
python transaction_monitoring_analysis.py

Charts will be saved as PNG files in the same folder.

---

## Parameters

Several parameters can be adjusted at the top of each cell without changing the rest of the code:

**`N_TOP_BANKS`** (default: 5): number of banks shown in Chart 1 (kpet low for readability)

**`N_NETWORK_BANKS`** (default: 10): number of banks shown in Chart 3 (kpet low for readability)

**`N_HEATMAP_BANKS`** (default: 10): number of banks shown in Chart 4 (kpet low for readability)

**`N_TRANSACTIONS`** (default: 200,000): number of synthetic transactions to generate. Can be increased for a larger dataset or decreased for faster execution.

**`START_DATE`** and **`END_DATE`**: define the date range of the analysis. Currently set to full year 2025:
```python
START_DATE = pd.Timestamp("2025-01-01")
END_DATE   = pd.Timestamp("2025-12-31")
```

---

## Data

All data is synthetically generated. No real bank data is used. The dataset simulates:
- 34 fictional banks with different trading profiles
- 200,000 transactions over 2025 business days
- Four instrument types (TYPE_A to TYPE_D)
- Deliberately injected data quality issues for demonstration purposes

---

## Dependencies

- Python 3.9+
- pandas
- numpy
- matplotlib