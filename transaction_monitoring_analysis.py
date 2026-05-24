"""
Transaction Monitoring Analysis
================================
This script simulates a simplified transaction monitoring workflow across
a portfolio of fictional banks operating within the ECB Single Supervisory
Mechanism (SSM).

Built as a portfolio project to demonstrate data processing, quality checking,
and visualisation skills in Python, as part of a job application to the
European Central Bank. Given that code developed in a professional context
is confidential, this project was created specifically to showcase relevant
technical skills using synthetic data.

Structure:
    1. Data Generation - synthetic raw transactions
    2. Data Quality - identify and remove problematic rows
    3. Data Compilation - aggregate to monthly level per bank
    4. Data visualisation - four charts
    5. Analysis - final statistical insights

Usage:
    python transaction_monitoring_analysis.py

Output:
    - Terminal output for quality checks and analysis summary
    - chart1_activity.png
    - chart2_scatter.png
    - chart3_network.png
    - chart4_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# Reproducibility: same synthetic data every time the script is run
np.random.seed(42)


# ==============================================================
# SECTION 1: DATA GENERATION
# ==============================================================

# Date range — full year 2025 (business days only)
START_DATE = pd.Timestamp("2025-01-01")
END_DATE   = pd.Timestamp("2025-12-31")
ALL_DATES  = pd.bdate_range(START_DATE, END_DATE)

# Number of banks per size category (fictional sizes)
# Large countries: 3 banks each (5 countries = 15 banks)
# Medium countries: 2 banks each (4 countries =  8 banks)
# Smaller countries: 1 bank each (11 countries = 11 banks)
N_BANKS_LARGE  = 5 * 3   # 15 banks
N_BANKS_MEDIUM = 4 * 2   # 8 banks
N_BANKS_SMALL = 11 * 1  # 11 banks
TOTAL_BANKS = N_BANKS_LARGE + N_BANKS_MEDIUM + N_BANKS_SMALL

# Naming the banks as BANK_XX
BANK_NAMES = [f"BANK_{i:02d}" for i in range(1, TOTAL_BANKS + 1)]

# Instrument types: generic labels
INSTRUMENT_TYPES = ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"]

# Price range per instrument type
PRICE_RANGES = {
    "TYPE_A": (1, 100),
    "TYPE_B": (10, 500),
    "TYPE_C": (50, 1_000),
    "TYPE_D": (100, 5_000),
}

# Bank trading profiles
# large: few trades, very high amounts (e.g. wholesale bank)
# frequent: many trades, small amounts (e.g. high-frequency trader)
# normal: balanced activity
BANK_PROFILES = {}
for i, bank in enumerate(BANK_NAMES):
    if i in [0, 1, 2]:
        BANK_PROFILES[bank] = "large"
    elif i in [3, 4, 5]:
        BANK_PROFILES[bank] = "frequent"
    else:
        BANK_PROFILES[bank] = "normal"


def generate_transactions(n_transactions: int = 200_000) -> pd.DataFrame:
    """
    Generate a synthetic dataset of raw transactions.

    Each row represents one transaction where:
    - The bank is always the BUYER
    - The counterparty is always the SELLER
    - A bank cannot trade with itself
    - Banks have uneven activity levels and different trading profiles:
        - large:    few trades, very high amounts
        - frequent: many trades, small amounts
        - normal:   balanced activity
    - AMOUNT_EUR = QUANTITY x PRICE_EUR

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate (default 200,000)

    Returns
    -------
    pd.DataFrame
        Raw transaction dataset with deliberate data quality issues injected.
    """

    n_banks = len(BANK_NAMES)

    # Assign uneven activity weights using exponential distribution
    raw_weights = np.random.exponential(scale=1.0, size=n_banks)
    activity_weights = raw_weights / raw_weights.sum()

    # Adjust weights by profile
    for i, bank in enumerate(BANK_NAMES):
        if BANK_PROFILES[bank] == "frequent":
            activity_weights[i] *= 3.0  # trade much more often
        elif BANK_PROFILES[bank] == "large":
            activity_weights[i] *= 0.3  # trade much less often

    # Renormalise after adjustment
    activity_weights = activity_weights / activity_weights.sum()

    # Sample buyer banks according to activity weights
    buyer_indices = np.random.choice(n_banks, size=n_transactions, p=activity_weights)

    # For each buyer, sample a counterparty (cannot be the same bank)
    seller_indices = np.array([
        np.random.choice([j for j in range(n_banks) if j != buyer_indices[i]])
        for i in range(n_transactions)
    ])

    # Random dates from business days
    date_indices = np.random.randint(0, len(ALL_DATES), n_transactions)

    # Random instrument types
    instrument_types = np.random.choice(INSTRUMENT_TYPES, n_transactions)

    # Quantity varies by bank profile
    quantities = np.array([
        np.random.randint(500_000, 1_000_000) if BANK_PROFILES[BANK_NAMES[buyer_indices[i]]] == "large"
        else np.random.randint(10, 500)        if BANK_PROFILES[BANK_NAMES[buyer_indices[i]]] == "frequent"
        else np.random.randint(100, 100_000)
        for i in range(n_transactions)
    ], dtype=float)

    # Price varies by instrument type
    prices = np.array([
        np.random.uniform(*PRICE_RANGES[inst])
        for inst in instrument_types
    ]).round(4)

    amounts = (quantities * prices).round(2)
    trade_ids = [f"TRD_{i:08d}" for i in range(n_transactions)]

    df = pd.DataFrame({
        "TRADE_ID": trade_ids,
        "DATE": ALL_DATES[date_indices],
        "BANK": [BANK_NAMES[i] for i in buyer_indices],
        "INSTRUMENT_TYPE": instrument_types,
        "COUNTERPARTY": [BANK_NAMES[i] for i in seller_indices],
        "SIDE": "Buy",
        "QUANTITY": quantities,
        "PRICE_EUR": prices,
        "AMOUNT_EUR": amounts,
    })

    # ----------------------------------------------------------
    # Inject deliberate data quality issues
    # ----------------------------------------------------------

    # Issue 1: 50 rows with null price (simulating a price feed failure)
    null_idx = np.random.choice(df.index, 50, replace=False)
    df.loc[null_idx, "PRICE_EUR"] = np.nan

    # Issue 2: 30 duplicate trade IDs (simulating double submission)
    dup_idx = np.random.choice(df.index[1:], 30, replace=False)
    df.loc[dup_idx, "TRADE_ID"] = df.loc[dup_idx - 1, "TRADE_ID"].values

    # Issue 3: 20 rows with negative amount (simulating a data entry error)
    neg_idx = np.random.choice(df.index, 20, replace=False)
    df.loc[neg_idx, "AMOUNT_EUR"] = -df.loc[neg_idx, "AMOUNT_EUR"]

    # Issue 4: 10 rows where amount != quantity x price (calculation error)
    calc_idx = np.random.choice(df.index, 10, replace=False)
    df.loc[calc_idx, "AMOUNT_EUR"] = (df.loc[calc_idx, "AMOUNT_EUR"] * 1.5).round(2)

    return df


# ==============================================================
# SECTION 2: DATA QUALITY CHECKS
# ==============================================================

def exploratory_summary(df: pd.DataFrame, label: str = "RAW DATA") -> None:
    """
    Print a summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarise
    label : str
        Label shown in the header
    """
    print("-" * 55)
    print(f"  EXPLORATORY SUMMARY - {label}")
    print("-" * 55)

    print(f"Total rows:              {len(df):>10,}")
    print(f"Unique banks:            {df['BANK'].nunique():>10,}")
    print(f"Unique counterparties:   {df['COUNTERPARTY'].nunique():>10,}")
    print(f"Unique instrument types: {df['INSTRUMENT_TYPE'].nunique():>10,}")
    print(f"Date range:              {df['DATE'].min().date()} to {df['DATE'].max().date()}")

    print("\n--- Transactions by Instrument Type ---")
    print(df["INSTRUMENT_TYPE"].value_counts().to_string())

    print("\n--- Transactions by Bank (activity distribution) ---")
    print(df["BANK"].value_counts().to_string())

    print("\n--- Amount (EUR) Statistics ---")
    print(df["AMOUNT_EUR"].describe().round(2).to_string())

    print("-" * 55)


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a series of data quality checks on the raw transaction dataset.

    Checks performed:
    1. Weekend transactions
    2. Missing values
    3. Duplicate trade IDs
    4. Negative or zero amount / price / quantity
    5. Sanity check: amount == quantity x price

    Each issue is reported and problematic rows are removed.
    Returns the cleaned DataFrame.
    """
    print("-" * 55)
    print("  DATA QUALITY CHECKS")
    print("-" * 55)

    rows_start = len(df)
    df = df.copy()

    # Check 1: Weekend transactions
    weekends   = df["DATE"].dt.dayofweek >= 5
    n_weekends = weekends.sum()
    if n_weekends > 0:
        print(f"WARNING - Weekend transactions: {n_weekends} rows found - removing affected rows.")
        df = df[~weekends]
    else:
        print("CHECK 1 PASSED - No transactions on weekends.")

    # Check 2: Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\nWARNING - Missing values found:")
        for col, n in missing.items():
            print(f"  {col}: {n} null values - removing affected rows.")
        df = df.dropna()
    else:
        print("CHECK 2 PASSED - No missing values.")

    # Check 3: Duplicate trade IDs
    duplicates = df.duplicated(subset=["TRADE_ID"])
    n_dupes    = duplicates.sum()
    if n_dupes > 0:
        print(f"\nWARNING - Duplicate trade IDs: {n_dupes} found - removing affected rows.")
        df = df[~duplicates]
    else:
        print("CHECK 3 PASSED - No duplicate trade IDs.")

    # Check 4: Negative or zero values
    invalid   = (df["AMOUNT_EUR"] <= 0) | (df["PRICE_EUR"] <= 0) | (df["QUANTITY"] <= 0)
    n_invalid = invalid.sum()
    if n_invalid > 0:
        print(f"\nWARNING - Negative or zero values: {n_invalid} rows found - removing affected rows.")
        df = df[~invalid]
    else:
        print("CHECK 4 PASSED - No negative or zero values.")

    # Check 5: Sanity check — amount == quantity x price
    tolerance       = 0.01
    expected_amount = (df["QUANTITY"] * df["PRICE_EUR"]).round(2)
    mismatch        = (df["AMOUNT_EUR"] - expected_amount).abs() > tolerance
    n_mismatch      = mismatch.sum()
    if n_mismatch > 0:
        print(f"\nWARNING - Amount mismatch: {n_mismatch} rows where AMOUNT_EUR != QUANTITY x PRICE_EUR - removing affected rows.")
        df = df[~mismatch]
    else:
        print("CHECK 5 PASSED - All amounts consistent with quantity x price.")

    rows_end = len(df)
    print(f"\nRows before cleaning: {rows_start:,}")
    print(f"Rows after cleaning:  {rows_end:,}")
    print(f"Rows removed:         {rows_start - rows_end:,}")
    print("-" * 55)

    return df.reset_index(drop=True)


# ==============================================================
# SECTION 3: DATA COMPILATION
# ==============================================================

def compile_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clean transaction data to monthly level per bank.

    For each bank and month, computes:
    - TOTAL_AMOUNT_EUR: sum of all traded amounts
    - TRADE_COUNT: number of trades executed
    - UNIQUE_COUNTERPARTIES: number of distinct counterparties traded with
    - TOP_INSTRUMENT_TYPE: most frequently traded instrument type
    - MOM_CHANGE_PCT: month-over-month % change in total amount

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction-level dataset

    Returns
    -------
    pd.DataFrame
        Monthly summary with one row per bank per month
    """

    df = df.copy()
    df["MONTH"] = df["DATE"].dt.to_period("M")

    # Most traded instrument type per bank per month
    top_instrument = (
        df.groupby(["BANK", "MONTH", "INSTRUMENT_TYPE"])
        .size()
        .reset_index(name="COUNT")
        .sort_values("COUNT", ascending=False)
        .drop_duplicates(subset=["BANK", "MONTH"])
        [["BANK", "MONTH", "INSTRUMENT_TYPE"]]
        .rename(columns={"INSTRUMENT_TYPE": "TOP_INSTRUMENT_TYPE"})
    )

    # Aggregations
    monthly = (
        df.groupby(["BANK", "MONTH"])
        .agg(
            TOTAL_AMOUNT_EUR = ("AMOUNT_EUR",   "sum"),
            TRADE_COUNT = ("TRADE_ID",     "count"),
            UNIQUE_COUNTERPARTIES = ("COUNTERPARTY", "nunique"),
        )
        .reset_index()
    )

    # Merge in top instrument type
    monthly = monthly.merge(top_instrument, on=["BANK", "MONTH"], how="left")

    # Sort before computing month-over-month change
    monthly = monthly.sort_values(["BANK", "MONTH"]).reset_index(drop=True)

    # Month-over-month % change in total amount per bank
    monthly["MOM_CHANGE_PCT"] = (
        monthly.groupby("BANK")["TOTAL_AMOUNT_EUR"]
        .pct_change() * 100
    ).round(2)

    monthly["TOTAL_AMOUNT_EUR"] = monthly["TOTAL_AMOUNT_EUR"].round(2)

    return monthly


# ==============================================================
# SECTION 4: DATA VISUALISATION
# ==============================================================

# ECB brand colours
BLACK = "#000000"
ECB_BLUE = "#003299"
ECB_YELLOW = "#FFB400"
ECB_RED = "#FF4B00"
ECB_LIGHT_GREEN = "#65B800"
ECB_CYAN = "#00B1EA"
ECB_LIGHT = "#F0F5FF"

# Shared figure style
plt.rcParams.update({
    "figure.facecolor": ECB_LIGHT,
    "axes.facecolor":   ECB_LIGHT,
    "axes.edgecolor":   BLACK,
    "axes.labelcolor":  BLACK,
    "xtick.color":      BLACK,
    "ytick.color":      BLACK,
    "text.color":       BLACK,
    "axes.titleweight": "bold",
    "axes.titlecolor":  ECB_BLUE,
})

# Chart parameters — change these values to show more or fewer banks
N_TOP_BANKS = 5   # Chart 1: number of banks in the line chart
N_NETWORK_BANKS = 10  # Chart 3: number of banks in the network heatmap
N_HEATMAP_BANKS = 10  # Chart 4: number of banks in the MoM heatmap


def plot_chart1(df_monthly: pd.DataFrame) -> None:
    """
    Chart 1 — Monthly trading activity for the top N most active banks.
    Two panels: total amount traded and trade count over time.
    """
    top_banks = (
        df_monthly.groupby("BANK")["TOTAL_AMOUNT_EUR"]
        .sum()
        .nlargest(N_TOP_BANKS)
        .index.tolist()
    )

    df_top = df_monthly[df_monthly["BANK"].isin(top_banks)].copy()
    df_top["MONTH_DT"] = df_top["MONTH"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Monthly Trading Activity - Top {N_TOP_BANKS} Most Active Banks (2025)",
        fontsize=13, fontweight="bold", color=ECB_BLUE
    )

    bank_colours = [ECB_BLUE, ECB_YELLOW, ECB_RED, ECB_LIGHT_GREEN, ECB_CYAN]

    for bank, color in zip(top_banks, bank_colours):
        bank_data = df_top[df_top["BANK"] == bank].sort_values("MONTH_DT")

        ax1.plot(
            bank_data["MONTH_DT"],
            bank_data["TOTAL_AMOUNT_EUR"] / 1e9,
            marker="o", markersize=4, linewidth=1.5,
            label=bank, color=color,
        )
        ax2.plot(
            bank_data["MONTH_DT"],
            bank_data["TRADE_COUNT"],
            marker="o", markersize=4, linewidth=1.5,
            label=bank, color=color,
        )

    ax1.set_ylabel("Total Amount (EUR Billions)", color=BLACK)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
    ax1.legend(title="Bank", fontsize=8)
    ax1.set_title("Total Amount Traded", fontsize=10, pad=8)

    ax2.set_ylabel("Number of Trades", color=BLACK)
    ax2.set_xlabel("Month", color=BLACK)
    ax2.set_title("Trade Count", fontsize=10, pad=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("chart1_activity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved: chart1_activity.png")


def plot_chart2(df_monthly: pd.DataFrame) -> None:
    """
    Chart 2 — Scatter plot of trading amount vs frequency for all banks.
    Identifies banks with unusual profiles (high amount/low frequency etc.)
    """
    scatter_data = (
        df_monthly.groupby("BANK")
        .agg(
            AVG_MONTHLY_AMOUNT = ("TOTAL_AMOUNT_EUR", "mean"),
            AVG_MONTHLY_TRADES = ("TRADE_COUNT",      "mean"),
            TOTAL_AMOUNT       = ("TOTAL_AMOUNT_EUR", "sum"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 9))

    scatter = ax.scatter(
        scatter_data["AVG_MONTHLY_TRADES"],
        scatter_data["AVG_MONTHLY_AMOUNT"] / 1e9,
        c=scatter_data["TOTAL_AMOUNT"],
        cmap="Blues",
        s=120,
        edgecolors=ECB_BLUE,
        linewidths=0.5,
        zorder=3,
    )

    # Only label banks that stand out — more than 1 std dev from mean
    # in either amount or trade count dimension
    mean_amount = scatter_data["AVG_MONTHLY_AMOUNT"].mean()
    std_amount  = scatter_data["AVG_MONTHLY_AMOUNT"].std()
    mean_trades = scatter_data["AVG_MONTHLY_TRADES"].mean()
    std_trades  = scatter_data["AVG_MONTHLY_TRADES"].std()

    banks_to_label = set(
        scatter_data[
            (scatter_data["AVG_MONTHLY_AMOUNT"] > mean_amount + std_amount) |
            (scatter_data["AVG_MONTHLY_AMOUNT"] < mean_amount - std_amount) |
            (scatter_data["AVG_MONTHLY_TRADES"] > mean_trades + std_trades) |
            (scatter_data["AVG_MONTHLY_TRADES"] < mean_trades - std_trades)
        ]["BANK"].tolist()
    )

    for _, row in scatter_data.iterrows():
        if row["BANK"] not in banks_to_label:
            continue
        x = row["AVG_MONTHLY_TRADES"]
        y = row["AVG_MONTHLY_AMOUNT"] / 1e9
        x_offset = 8 if x < scatter_data["AVG_MONTHLY_TRADES"].median() else -50
        ax.annotate(
            row["BANK"], (x, y),
            fontsize=7, xytext=(x_offset, 6),
            textcoords="offset points", color=ECB_BLUE,
            arrowprops=dict(arrowstyle="-", color=ECB_BLUE, alpha=0.3, lw=0.5),
        )

    median_trades = scatter_data["AVG_MONTHLY_TRADES"].median()
    median_amount = (scatter_data["AVG_MONTHLY_AMOUNT"] / 1e9).median()

    ax.axvline(median_trades, color=ECB_CYAN, linestyle="--",
               linewidth=1, label=f"Median trades: {median_trades:.0f}")
    ax.axhline(median_amount, color=ECB_RED, linestyle="--",
               linewidth=1, label=f"Median amount: {median_amount:.1f}B")

    x_min = scatter_data["AVG_MONTHLY_TRADES"].min()
    x_max = scatter_data["AVG_MONTHLY_TRADES"].max()
    y_min = (scatter_data["AVG_MONTHLY_AMOUNT"] / 1e9).min()
    y_max = (scatter_data["AVG_MONTHLY_AMOUNT"] / 1e9).max()

    ax.text(x_min, y_max * 0.95, "High amount / Low frequency",
            fontsize=8, color=ECB_BLUE, alpha=0.6, va="top")
    ax.text(x_max * 0.65, y_max * 0.95, "High amount / High frequency",
            fontsize=8, color=ECB_BLUE, alpha=0.6, va="top")
    ax.text(x_min, y_min * 2.5, "Low amount / Low frequency",
            fontsize=8, color=ECB_BLUE, alpha=0.6, va="bottom")
    ax.text(x_max * 0.65, y_min * 2.5, "Low amount / High frequency",
            fontsize=8, color=ECB_BLUE, alpha=0.6, va="bottom")

    plt.colorbar(scatter, ax=ax, label="Total Annual Amount (EUR)")
    ax.set_xlabel("Average Monthly Trade Count", color=BLACK)
    ax.set_ylabel("Average Monthly Amount (EUR Billions)", color=BLACK)
    ax.set_title("Trading Amount vs Frequency per Bank - All Banks (2025)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("chart2_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved: chart2_scatter.png")


def plot_chart3(df_clean: pd.DataFrame) -> None:
    """
    Chart 3 — Heatmap matrix showing number of trades between bank pairs.
    Reveals counterparty concentration and network structure.
    """
    top_banks_network = (
        df_clean.groupby("BANK")["TRADE_ID"]
        .count()
        .nlargest(N_NETWORK_BANKS)
        .index.tolist()
    )

    df_network = df_clean[
        df_clean["BANK"].isin(top_banks_network) &
        df_clean["COUNTERPARTY"].isin(top_banks_network)
    ]

    network_matrix = (
        df_network.groupby(["BANK", "COUNTERPARTY"])["TRADE_ID"]
        .count()
        .unstack(fill_value=0)
    )
    network_matrix = network_matrix.reindex(
        index=top_banks_network, columns=top_banks_network, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(network_matrix.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(top_banks_network)))
    ax.set_xticklabels(top_banks_network, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_banks_network)))
    ax.set_yticklabels(top_banks_network, fontsize=9)

    # Dynamic text colour: white on dark cells, ECB blue on light cells
    vmin      = network_matrix.values[network_matrix.values > 0].min()
    vmax      = network_matrix.values.max()
    threshold = (vmax - vmin) / 2

    for i in range(len(top_banks_network)):
        for j in range(len(top_banks_network)):
            val = network_matrix.values[i, j]
            if val > 0:
                text_color = "white" if val > threshold else ECB_BLUE
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Number of Trades")
    ax.set_title(f"Counterparty Network - Number of Trades Between Bank Pairs (Top {N_NETWORK_BANKS} Banks)")
    ax.set_xlabel("Counterparty (Seller)", color=BLACK)
    ax.set_ylabel("Bank (Buyer)", color=BLACK)

    plt.tight_layout()
    plt.savefig("chart3_network.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved: chart3_network.png")


def plot_chart4(df_monthly: pd.DataFrame) -> None:
    """
    Chart 4 — Heatmap of month-over-month change in trading amount.
    Shows which banks have the most volatile activity over time.
    """
    top_banks_heatmap = (
        df_monthly.groupby("BANK")["TOTAL_AMOUNT_EUR"]
        .sum()
        .nlargest(N_HEATMAP_BANKS)
        .index.tolist()
    )

    heatmap_data = (
        df_monthly[df_monthly["BANK"].isin(top_banks_heatmap)]
        .pivot_table(index="BANK", columns="MONTH", values="MOM_CHANGE_PCT")
    )
    heatmap_data.columns = [str(c) for c in heatmap_data.columns]

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(heatmap_data.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)

    # Dynamic text colour: white on dark cells, ECB blue on light cells
    vmin_h      = np.nanmin(heatmap_data.values)
    vmax_h      = np.nanmax(heatmap_data.values)
    threshold_h = (vmax_h - vmin_h) / 2

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > threshold_h else ECB_BLUE
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=6.5, color=text_color)

    plt.colorbar(im, ax=ax, label="Month-over-Month Change (%)")
    ax.set_title(f"Month-over-Month Change in Trading Amount - Top {N_HEATMAP_BANKS} Banks (2025)")
    ax.set_xlabel("Month", color=BLACK)
    ax.set_ylabel("Bank", color=BLACK)

    plt.tight_layout()
    plt.savefig("chart4_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved: chart4_heatmap.png")


# ==============================================================
# SECTION 5: ANALYSIS
# ==============================================================

def print_analysis(df_monthly: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Print key statistical insights from the compiled monthly dataset.

    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly compiled dataset
    df_clean : pd.DataFrame
        Clean transaction-level dataset
    """
    print("-" * 55)
    print("  ANALYSIS SUMMARY")
    print("-" * 55)

    # Most and least active banks by total annual amount
    annual = df_monthly.groupby("BANK")["TOTAL_AMOUNT_EUR"].sum()
    print(f"Most active bank: {annual.idxmax()} ({annual.max()/1e9:.2f}B EUR)")
    print(f"Least active bank: {annual.idxmin()} ({annual.min()/1e9:.2f}B EUR)")
    print(f"Activity ratio (most/least): {annual.max()/annual.min():.1f}x")

    # Most concentrated bank (fewest avg counterparties)
    conc = df_monthly.groupby("BANK")["UNIQUE_COUNTERPARTIES"].mean()
    print(f"\nMost concentrated bank:  {conc.idxmin()} ({conc.min():.1f} avg counterparties/month)")
    print(f"Least concentrated bank: {conc.idxmax()} ({conc.max():.1f} avg counterparties/month)")

    # Most volatile bank (highest std dev of MoM change)
    vol = df_monthly.groupby("BANK")["MOM_CHANGE_PCT"].std().dropna()
    print(f"\nMost volatile bank:  {vol.idxmax()} (std dev: {vol.max():.1f}%)")
    print(f"Least volatile bank: {vol.idxmin()} (std dev: {vol.min():.1f}%)")

    # Most common top instrument type
    top_inst = df_monthly["TOP_INSTRUMENT_TYPE"].value_counts().idxmax()
    print(f"\nMost common top instrument type: {top_inst}")

    # Average MoM change across all banks
    avg_mom = df_monthly["MOM_CHANGE_PCT"].mean()
    print(f"Average MoM change across all banks: {avg_mom:.2f}%")

    # Most active trading pair — links to Chart 3
    pair_counts = (
        df_clean.groupby(["BANK", "COUNTERPARTY"])["TRADE_ID"]
        .count()
        .reset_index()
        .rename(columns={"TRADE_ID": "TRADE_COUNT"})
        .sort_values("TRADE_COUNT", ascending=False)
        .iloc[0]
    )
    print(f"\nMost active trading pair: {pair_counts['BANK']} -> {pair_counts['COUNTERPARTY']} ({pair_counts['TRADE_COUNT']:,} trades)")

    # Most active month — links to Chart 4
    monthly_total = df_monthly.groupby("MONTH")["TOTAL_AMOUNT_EUR"].sum()
    best_month    = monthly_total.idxmax()
    print(f"Most active month: {best_month} (total volume: {monthly_total.max()/1e9:.1f}B EUR)")

    print("-" * 55)


# ==============================================================
# MAIN — run all sections in order
# ==============================================================

if __name__ == "__main__":

    # Section 1: Generate data
    print("\n" + "=" * 55)
    print("  SECTION 1 - DATA GENERATION")
    print("=" * 55)
    print(f"Total banks: {len(BANK_NAMES)}")
    df_raw = generate_transactions()
    print(f"Dataset generated: {len(df_raw):,} transactions")
    print(f"Date range: {df_raw['DATE'].min().date()} to {df_raw['DATE'].max().date()}")

    # Section 2: Quality checks
    print("\n" + "=" * 55)
    print("  SECTION 2 - DATA QUALITY CHECKS")
    print("=" * 55)
    exploratory_summary(df_raw, label="RAW DATA")
    df_clean = run_quality_checks(df_raw)
    exploratory_summary(df_clean, label="CLEAN DATA")

    # Section 3: Compile monthly
    print("\n" + "=" * 55)
    print("  SECTION 3 - DATA COMPILATION")
    print("=" * 55)
    df_monthly = compile_monthly(df_clean)
    print(f"Monthly dataset compiled: {len(df_monthly):,} rows")
    print(f"Banks: {df_monthly['BANK'].nunique()} | Months: {df_monthly['MONTH'].nunique()}")

    # Section 4: Visualisation
    print("\n" + "=" * 55)
    print("  SECTION 4 - VISUALISATION")
    print("=" * 55)
    plot_chart1(df_monthly)
    plot_chart2(df_monthly)
    plot_chart3(df_clean)
    plot_chart4(df_monthly)

    # Section 5: Analysis
    print("\n" + "=" * 55)
    print("  SECTION 5 - ANALYSIS")
    print("=" * 55)
    print_analysis(df_monthly, df_clean)
