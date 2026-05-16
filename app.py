import pandas as pd
# Raw text → TfidfVectorizer → numbers → MultinomialNB → category prediction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# read the csv file, using df is convention 
df = pd.read_csv("transactions.csv", skipinitialspace=True, skiprows=1)
df = df.drop(columns=[df.columns[0]])

# rename columns to be easier to type
df.columns = ["type", "date", "amount", "description"]

# convert date column to a proper date format
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# convert amount to a number in case it loaded as text
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# keep only spending rows (negative amounts = money leaving your account)
spending = df[df["amount"] < 0].copy()

# make all amounts positive so they're easier to read and work with
spending["amount"] = spending["amount"].abs()

# ── STEP 1: manually label known transactions to train the classifier ──
# format is: "keyword in description" : "category"
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

# ── STEP 2: assign labels to matching rows ──
def assign_label(description):
    description = description.upper()  # uppercase so matching works regardless of capitalization
    for keyword, category in labels.items():
        if keyword in description:
            return category  # return category immediately and stop checking
    return None  # if no keyword matched, return None (unlabelled)

# run assign_label on every row and store results in a new category column
spending["category"] = spending["description"].apply(assign_label)

# ── STEP 3: split into labelled and unlabelled ──
labelled = spending[spending["category"].notna()]   # rows where a keyword matched
unlabelled = spending[spending["category"].isna()]  # rows where no keyword matched

# ── STEP 4: train the classifier on the labelled transactions ──
# TfidfVectorizer converts description text into numbers the model can read
vectorizer = TfidfVectorizer()

# fit_transform learns the vocabulary and converts labelled descriptions to numbers
X_train = vectorizer.fit_transform(labelled["description"])

# these are the correct answers we're training the model on
y_train = labelled["category"]

# create the classifier and train it on our labelled data
model = MultinomialNB()
model.fit(X_train, y_train)

# ── STEP 5: predict categories for unlabelled transactions ──
# transform converts unlabelled descriptions using the same vocabulary
X_unlabelled = vectorizer.transform(unlabelled["description"])

# predict runs each unlabelled transaction through the trained model
predicted_categories = model.predict(X_unlabelled)

# store the predictions back in the spending dataframe
spending.loc[unlabelled.index, "category"] = predicted_categories