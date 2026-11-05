# retail-demand-forecasting
# Retail Demand Forecasting & Stock Recommendation
🇹🇷 [Türkçe versiyon](README_TR.md)
Forecasting daily item-level sales across 10 stores and 50 items, and turning the forecast into weekly stock recommendations that are backtested against actual demand.

**Result:** a log-linear regression model reduces forecast error by **30%** versus a seasonal-naive baseline (MAE 8.35 → 5.85), and the resulting stock recommendation covers demand in **98.3%** of store-item-weeks against a 95% target.

---

## Problem

A retailer needs to know how many units of each item to stock in each store. Too little stock means empty shelves and lost sales; too much means waste and capital tied up in inventory. This project answers two questions:

1. **Forecasting:** How many units of each item will each store sell per day?
2. **Decision:** How much stock should each store hold per week to meet demand reliably?

## Data

- **Source:** [Kaggle – Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) (`train.csv`)
- **Size:** 913,000 rows — 10 stores × 50 items × daily sales, 2013-01-01 to 2017-12-31
- **Fields:** `date`, `store`, `item`, `sales`

The data is not included in this repository (competition rules). Download `train.csv` from Kaggle and place it next to the notebook.

## Approach

### 1. Data profiling
- No missing values; complete daily history for every store-item pair.
- Three clear patterns:
  - **Weekly cycle:** sales rise steadily through the week; Sunday is ~50% higher than Monday.
  - **Yearly seasonality:** low in January, peak in July, drop in December.
  - **Trend:** average daily sales grew from ~43.5 (2013) to ~59 (2017).

### 2. Evaluation setup
- **Test period:** last 3 months (Oct–Dec 2017, 46,000 rows), held out from all modelling decisions.
- **Metric:** Mean Absolute Error (MAE), in units per store-item-day.
- All parameters (growth rate, error estimates) are computed **only from pre-test data** to avoid leakage.

### 3. Forecasting methods

| Method | Description | Test MAE |
|---|---|---|
| Seasonal naive (baseline) | Same weekday, 52 weeks earlier | 8.35 |
| + Trend adjustment | Baseline × year-over-year growth (3.9%) | 8.31 |
| Seasonal profile | 2016 average by store × item × month × weekday, × growth | 6.41 |
| **Log-linear regression** | Linear regression on `log(sales)` with store, item, weekday, month (one-hot) and year (trend) | **5.85** |

**Why log scale?** Weekday and seasonal effects are multiplicative (e.g. Sunday ≈ ×1.5), not additive. Modelling `log(sales)` lets a linear model learn percentage effects.

### 4. Stock recommendation
- Weekly recommended stock = weekly forecast + safety stock
- Safety stock = 1.65 × daily forecast error std × √(days in week) → targets a **95% service level**
- Forecast error std is estimated per store-item on Jul–Sep 2017 (pre-test).

**Backtest:** recommended stock covered actual weekly demand in **98.3%** of store-item-weeks.

## Key findings

- The trend adjustment fixed the baseline's systematic under-forecasting (mean forecast 52.4 → 54.5 vs. actual 54.7), but barely moved MAE — most baseline error comes from single-day noise, not level bias.
- Averaging over similar days (same month and weekday) removes much of that noise: MAE 8.31 → 6.41.
- **Debugging note:** the weekday feature was initially computed as day-of-month, which raised MAE to 10.74. Checking row counts, nulls and sample rows revealed the issue; fixing it brought MAE to 6.41.
- The stock policy is slightly conservative (98.3% vs. 95% target), likely because error was estimated on high-volume summer months while the test period has lower volume.

## Limitations & next steps

- Calibrate safety stock by season, so the service level lands closer to target and excess inventory is reduced.
- Try gradient boosting (e.g. LightGBM) with lag and rolling-window features.
- Compare forecast accuracy by item volume (fast vs. slow movers).
- Add a simple what-if view: stock levels under different service-level targets.

## How to run

1. Download `train.csv` from the Kaggle link above.
2. Open `demand_forecasting.ipynb` in Google Colab or Jupyter.
3. Upload `train.csv` and run all cells.

**Libraries:** pandas, numpy, matplotlib, scikit-learn

## AI-assisted workflow

This project was built with Claude as an AI pair programmer. The AI drafted code and explanations; I ran every step, checked the outputs against expectations, and questioned results that did not make sense. The day-of-month bug above is an example: the result looked wrong, so I ran diagnostic checks, traced the root cause and corrected it before moving on.

## Author

**Miray Hilesiz** — MSc Management Information Systems, Haliç University
[LinkedIn](https://linkedin.com/in/mirayhilesiz) · [GitHub](https://github.com/hilesizmiray-cmyk) · [Website](https://mirayhilesiz.com)
