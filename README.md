# Marketing performance explorer

A Streamlit app for marketing analysts. Upload a campaign CSV, check data quality,
see key metrics and charts, and find out whether a machine-learning model can predict
conversion rate, revenue or ROI better than simply guessing the average.

## Run it (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Your CSV needs these columns

`Clicks`, `Conversions`, `Budget`, `Revenue_Generated`, `Units_Sold`, `ROI`.
Optional: `Subscription_Tier`, `Common_Keywords`, `Subscription_Length`, `Discount_Level`.
Columns ending in `_ID` are ignored by the models.

## What it does

- **Overview:** rows uploaded, removed (duplicates, conversions above clicks, negatives) and kept.
- **KPIs & charts:** headline metrics, group charts, scatter plot, correlation heatmap.
- **Model comparison:** linear regression, random forest and gradient boosting vs a baseline, using 5-fold cross-validation.
- **Predict:** a what-if form with the result shown against the dataset average.
