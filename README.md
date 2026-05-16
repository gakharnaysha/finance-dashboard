# Personal Finance Dashboard

Tired of not knowing where my money was going, so I built this. Upload your BMO transaction CSV and it automatically sorts everything into categories and shows you a breakdown.

## Demo
[Live app]: https://finance-dashboard-ek8ndya3kuksgyrdgw9r2z.streamlit.app/

## What it does
- Categorizes transactions into Food & Drink, Shopping, Transport, Subscriptions, etc.
- Shows total spent, biggest category, and a pie chart breakdown
- Lets you filter transactions by category
- Shows the biggest purchase in each category

## How the categorization works
Known merchants get matched by keyword. Anything unrecognized gets classified by a Naive Bayes model trained on the labelled transactions. Not perfect but works pretty well.

## Stack
Python, Pandas, scikit-learn, Streamlit, Matplotlib

## Running locally
```bash
git clone https://github.com/gakharnaysha/finance-dashboard.git
cd finance-dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```
Then upload your BMO CSV in the browser.

## Exporting from BMO
Login → select your account → Download Transactions → set date range → CSV format
