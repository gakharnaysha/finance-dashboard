import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app import spending

# ── PAGE CONFIG ──
st.set_page_config(page_title="Finance Dashboard", layout="centered")

# ── HEADER ──
st.title("Personal Finance Dashboard")
st.markdown("Spending analysis based on your BMO transactions.")


# ── EXCLUDE TRANSFERS ──
# transfers skew totals since they include rent and tuition
actual_spending = spending[spending["category"] != "Transfer"]

# ── DATE RANGE ──
date_min = actual_spending["date"].min().strftime("%B %d, %Y")
date_max = actual_spending["date"].max().strftime("%B %d, %Y")
num_months = round((actual_spending["date"].max() - actual_spending["date"].min()).days / 30)
st.caption(f"Tracking {num_months} months of transactions — {date_min} to {date_max}")

# ── SUMMARY METRICS ──
total_spent = actual_spending["amount"].sum()
num_transactions = len(actual_spending)
top_category = actual_spending.groupby("category")["amount"].sum().idxmax()
biggest_purchase = actual_spending.loc[actual_spending["amount"].idxmax(), "description"]
biggest_purchase_amount = actual_spending["amount"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"${total_spent:,.2f}")
col2.metric("Transactions", num_transactions)
col3.metric("Biggest Category", top_category)

st.divider()

# ── CATEGORY BREAKDOWN TABLE ──
st.subheader("Spending by Category")

# build a summary table with amount, transaction count and percentage
category_summary = actual_spending.groupby("category")["amount"].agg(["sum", "count"])
category_summary.columns = ["Total Spent", "Transactions"]
category_summary["% of Spending"] = (category_summary["Total Spent"] / total_spent * 100).round(1).astype(str) + "%"
category_summary["Total Spent"] = category_summary["Total Spent"].apply(lambda x: f"${x:,.2f}")
category_summary = category_summary.sort_values("Transactions", ascending=False)

st.dataframe(category_summary, use_container_width=True)

st.divider()

# ── PIE CHART ──
st.subheader("Category Breakdown")

pie_data = actual_spending.groupby("category")["amount"].sum()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    pie_data,
    labels=pie_data.index,
    autopct="%1.1f%%",  # shows percentage on each slice
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
ax.set_title("Where your money goes", fontsize=13)
st.pyplot(fig)

st.divider()

# ── BIGGEST PURCHASE PER CATEGORY ──
st.subheader("Biggest Purchase per Category")

# for each category find the single most expensive transaction
biggest_per_category = actual_spending.loc[actual_spending.groupby("category")["amount"].idxmax()]
biggest_per_category = biggest_per_category[["category", "description", "amount", "date"]].sort_values("amount", ascending=False)
biggest_per_category["amount"] = biggest_per_category["amount"].apply(lambda x: f"${x:,.2f}")
biggest_per_category = biggest_per_category.rename(columns={
    "category": "Category",
    "description": "Description",
    "amount": "Amount",
    "date": "Date"
})

st.dataframe(biggest_per_category.set_index("Category"), use_container_width=True)

st.divider()

# ── FULL TRANSACTION LIST ──
st.subheader("All Transactions")

# let user filter by category
categories = ["All"] + sorted(actual_spending["category"].unique().tolist())
selected = st.selectbox("Filter by category", categories)

# show filtered or full list depending on selection
if selected == "All":
    filtered = actual_spending
else:
    filtered = actual_spending[actual_spending["category"] == selected]

# format for display
display = filtered[["date", "description", "amount", "category"]].copy()
display["amount"] = display["amount"].apply(lambda x: f"${x:,.2f}")
display = display.rename(columns={
    "date": "Date",
    "description": "Description",
    "amount": "Amount",
    "category": "Category"
}).sort_values("Date", ascending=False)

st.dataframe(display, use_container_width=True)