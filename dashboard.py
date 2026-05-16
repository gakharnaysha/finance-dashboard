import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ── PAGE CONFIG ──
st.set_page_config(page_title="Finance Dashboard", layout="centered")
st.title("💳 Personal Finance Dashboard")
st.markdown("Spending analysis based on your BMO transactions.")

# ── FILE UPLOADER ──
uploaded_file = st.file_uploader("Upload your BMO transactions CSV", type="csv")

if uploaded_file is None:
    st.info("Export your transactions from BMO Online Banking and upload the CSV above to get started.")
    st.stop()  # stops the rest of the app from running until a file is uploaded

# ── LOAD AND CLEAN DATA ──
df = pd.read_csv(uploaded_file, skipinitialspace=True, skiprows=1)
df = df.drop(columns=[df.columns[0]])
df.columns = ["type", "date", "amount", "description"]
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
spending = df[df["amount"] < 0].copy()
spending["amount"] = spending["amount"].abs()

# ── LABELS ──
labels = {
    "DOLLARAMA": "Shopping",
    "TF 0489": "Transfer",
    "IQBAL FOODS": "Groceries",
    "WINNERS": "Shopping",
    "TIM HORTONS": "Food & Drink",
    "BAO HOUSE": "Food & Drink",
    "TF 0004510430": "Transfer",
    "MAGPIE": "Food & Drink",
    "MCMASTER HOSPITALITY": "Food & Drink",
    "SHOPPERS WORLD": "Shopping",
    "HONK": "Transport",
    "INTERAC ETRNSFR": "Transfer",
    "BLUENOTES": "Shopping",
    "SPOTIFY": "Subscriptions",
    "HOLLISTER": "Shopping",
    "SHOWCASE": "Entertainment",
    "CDNCANCERSOC": "Other",
    "SQ *": "Food & Drink",
    "UBER": "Transport",
    "LYFT": "Transport",
    "REXALL": "Health",
    "SHOPPERS DRUG": "Health",
    "MANSION": "Entertainment",
    "CALL IT SPRING": "Shopping",
    "ART GALLERY": "Entertainment",
    "GOOGLE": "Subscriptions",
    "SEPHORA": "Shopping",
    "CRUMBL": "Food & Drink",
    "EAST SIDE MARIO": "Food & Drink",
    "BASKIN ROBBINS": "Food & Drink",
    "CLAUDE.AI": "Subscriptions",
    "DOMINOS": "Food & Drink",
    "BOOSTER JUICE": "Food & Drink",
    "CANADIANCANCERSOC": "Other",
    "TNA": "Shopping",
    "MICHAELS": "Shopping",
}

def assign_label(description):
    description = description.upper()
    for keyword, category in labels.items():
        if keyword in description:
            return category
    return None

spending["category"] = spending["description"].apply(assign_label)

labelled = spending[spending["category"].notna()]
unlabelled = spending[spending["category"].isna()]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(labelled["description"])
y_train = labelled["category"]
model = MultinomialNB()
model.fit(X_train, y_train)

X_unlabelled = vectorizer.transform(unlabelled["description"])
predicted_categories = model.predict(X_unlabelled)
spending.loc[unlabelled.index, "category"] = predicted_categories

# ── EXCLUDE TRANSFERS ──
actual_spending = spending[spending["category"] != "Transfer"]

# ── DATE RANGE ──
date_min = actual_spending["date"].min().strftime("%B %d, %Y")
date_max = actual_spending["date"].max().strftime("%B %d, %Y")
num_months = round((actual_spending["date"].max() - actual_spending["date"].min()).days / 30)
st.caption(f"📅 Tracking {num_months} months of transactions — {date_min} to {date_max}")

# ── SUMMARY METRICS ──
total_spent = actual_spending["amount"].sum()
num_transactions = len(actual_spending)
top_category = actual_spending.groupby("category")["amount"].sum().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"${total_spent:,.2f}")
col2.metric("Transactions", num_transactions)
col3.metric("Biggest Category", top_category)

st.divider()

# ── CATEGORY BREAKDOWN TABLE ──
st.subheader("📊 Spending by Category")
category_summary = actual_spending.groupby("category")["amount"].agg(["sum", "count"])
category_summary.columns = ["Total Spent", "Transactions"]
category_summary["% of Spending"] = (category_summary["Total Spent"] / total_spent * 100).round(1).astype(str) + "%"
categor