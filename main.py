import csv
import ipaddress
import json
import os
import re
import socket
import time
import unicodedata
from collections import defaultdict, deque
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- security settings (set these on the hosting service; leave empty on your laptop) ----------
# FIREBASE_PROJECT_ID: turns on login checks. Empty = no login needed (local testing).
# ALLOWED_ORIGINS: website addresses allowed to call this server, comma-separated.
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))

app = FastAPI(title="TruPick API", docs_url=None if FIREBASE_PROJECT_ID else "/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

if FIREBASE_PROJECT_ID:
    import cachecontrol
    import google.auth.transport.requests
    from google.oauth2 import id_token as google_id_token
    _google_request = google.auth.transport.requests.Request(session=cachecontrol.CacheControl(requests.Session()))

_recent = defaultdict(deque)


def require_user(authorization: str = Header(default="")):
    """Checks the Firebase login token sent by the website. Returns the user's id."""
    if not FIREBASE_PROJECT_ID:
        return "local"
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in to use TruPick.")
    try:
        claims = google_id_token.verify_firebase_token(authorization[7:], _google_request, audience=FIREBASE_PROJECT_ID)
    except Exception:
        raise HTTPException(401, "Your sign-in has expired. Please sign in again.")
    if not claims or claims.get("iss") != f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}":
        raise HTTPException(401, "Please sign in to use TruPick.")
    provider = (claims.get("firebase") or {}).get("sign_in_provider")
    if provider == "password" and not claims.get("email_verified"):
        raise HTTPException(403, "Please verify your email first. Check your inbox for the link.")
    uid = claims.get("user_id") or claims.get("sub")
    # simple rate limit per user
    now, q = time.time(), _recent[uid]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many checks in a minute. Please wait a moment.")
    q.append(now)
    return uid

# Load the review-based sentiment model
model = joblib.load('quality_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# One similarity index per category (a category is skipped if its files are missing)
CATEGORIES = {
    "womens_clothing": ("Women's clothing", "product_vectorizer.pkl", "product_vectors.pkl", "product_profiles.csv"),
    "electronics": ("Electronics", "electronics_vectorizer.pkl", "electronics_vectors.pkl", "electronics_profiles.csv"),
    "beauty": ("Beauty & personal care", "beauty_vectorizer.pkl", "beauty_vectors.pkl", "beauty_profiles.csv"),
    "fashion_footwear": ("Fashion & footwear", "fashion_footwear_vectorizer.pkl", "fashion_footwear_vectors.pkl", "fashion_footwear_profiles.csv"),
    "home_kitchen": ("Home & kitchen", "home_kitchen_vectorizer.pkl", "home_kitchen_vectors.pkl", "home_kitchen_profiles.csv"),
    "phones_accessories": ("Phones & accessories", "phones_accessories_vectorizer.pkl", "phones_accessories_vectors.pkl", "phones_accessories_profiles.csv"),
    "sports_fitness": ("Sports & fitness", "sports_fitness_vectorizer.pkl", "sports_fitness_vectors.pkl", "sports_fitness_profiles.csv"),
    "health_household": ("Health & household", "health_household_vectorizer.pkl", "health_household_vectors.pkl", "health_household_profiles.csv"),
}

# "What buyers say" files (optional): same prefix + insights.json
def insights_file(vec_file):
    return vec_file.replace("vectorizer.pkl", "insights.json")


# Advice for "Check before you buy", keyed by the aspect buyers complain about
ADVICE = {
    "Battery life": "Check recent reviews for real battery hours, not just the advertised number.",
    "Charging": "Look for complaints about slow or failed charging, and check the cable/port type.",
    "Sound quality": "Look for reviews mentioning low volume or weak bass at this price.",
    "Connectivity": "Check reviews for dropped Bluetooth connections or one side not connecting.",
    "Microphone & calls": "If you take calls, look for reviews about mic quality in noisy places.",
    "Screen & display": "Check brightness and how the screen looks outdoors in buyer photos.",
    "Fit & size": "Check the size chart and reviews saying it runs small or large.",
    "Comfort": "Look for reviews from people who used it for hours, not minutes.",
    "Build quality": "Read the 1\u20132 star reviews for things that broke within weeks.",
    "Material & fabric": "Zoom into buyer photos to judge the material, thickness and stitching.",
    "Look & colour": "Compare buyer photos with seller photos; colours often differ.",
    "Value for money": "Compare prices of similar products; some buyers felt it was overpriced.",
    "Scent": "Check reviews about the smell, especially if you are sensitive to fragrance.",
    "Skin feel": "Check reviews from people with your skin type (oily or dry).",
    "Hair results": "Look for reviews from people with your hair type.",
    "Irritation": "Do a patch test first; some buyers reported irritation or breakouts.",
    "Long-lasting": "Check how long it lasts in real use, not just on day one.",
    "Easy to use": "Read reviews about setup or assembly before buying.",
    "Cleaning": "Check how easy it is to clean and whether it stains.",
    "Leaks": "Look for reviews mentioning leaks, especially for bottles and bags.",
    "Grip": "Check reviews about slipping in real use.",
    "Weight": "Check the actual weight; some buyers found it heavier or lighter than expected.",
    "Effectiveness": "Look for reviews from people who used it for a few weeks, not days.",
    "Taste": "Check reviews about taste and aftertaste.",
    "Protection": "Check that it really fits your exact model and read drop-test reviews.",
    "Packaging & delivery": "Check recent reviews about damaged or opened packages.",
}
# Category-specific names and tips: (shown name, tip). Anything not listed uses ADVICE above.
CATEGORY_WORDING = {
    "home_kitchen": {
        "Fit & size": ("Size & capacity", "Check the exact size or capacity (ml / cm) in the listing; buyers often find it smaller than the photos."),
        "Material & fabric": ("Material", "Check whether it's glass, steel or plastic, and look at buyer photos for thickness."),
        "Build quality": ("Durability", "Look for 1\u20132 star reviews about cracks, chips, rust or broken lids after a few weeks."),
        "Effectiveness": ("Does the job", "Read reviews about everyday use, not just first impressions."),
        "Look & colour": ("Look & finish", "Compare buyer photos with seller photos; finish and colour often differ."),
        "Comfort": ("Comfort", "For bedding or cushions, look for reviews after a few weeks of use."),
        "Leaks": ("Leaks & seal", "For bottles and containers, check reviews about leaking lids or loose seals."),
        "Cleaning": ("Easy to clean", "Check if it's dishwasher-safe and whether it stains or holds smells."),
    },
    "phones_accessories": {
        "Fit & size": ("Fit for your phone", "Make sure it's made for your exact model (e.g. iPhone 13 vs 13 Pro); cut-outs often don't line up."),
        "Build quality": ("Durability", "Look for reviews about cables fraying, cases cracking or yellowing."),
        "Material & fabric": ("Material", "Check if it's silicone, TPU or hard plastic, and how it feels in buyer photos."),
        "Grip": ("Grip", "Check reviews about the case being slippery in hand."),
        "Charging": ("Charging speed", "Check the wattage and whether buyers got fast charging with their phone."),
    },
    "sports_fitness": {
        "Fit & size": ("Size & fit", "Check the size chart; for mats and gear, check thickness and dimensions too."),
        "Build quality": ("Durability", "Look for reviews about tearing, peeling or breaking after regular workouts."),
        "Material & fabric": ("Material", "Check the material and whether buyers mention a strong smell when new."),
        "Grip": ("Grip", "Check reviews about slipping when sweaty."),
    },
    "health_household": {
        "Effectiveness": ("Results", "Look for reviews from people who used it for a few weeks; results take time."),
        "Build quality": ("Durability", "For devices, look for reviews about them stopping working after a few months."),
        "Easy to use": ("Easy to use", "Check reviews about pill size, dosage or how easy the device is to use."),
    },
    "electronics": {
        "Fit & size": ("Fit", "For earbuds or headphones, look for reviews about fit for small or large ears."),
        "Build quality": ("Reliability", "Read 1\u20132 star reviews for things that stopped working within weeks."),
        "Comfort": ("Comfort", "Look for reviews from people who wore or used it for hours."),
    },
    "beauty": {
        "Effectiveness": ("Results", "Look for reviews after a few weeks of use, from people with your skin or hair type."),
        "Look & colour": ("Shade & colour", "Compare the shade in buyer photos taken in daylight."),
        "Long-lasting": ("Lasts all day", "Check reviews about how long it stays on, not just how it looks at first."),
    },
}
CATEGORY_GENERAL = {
    "home_kitchen": "Check the return policy for breakable or kitchen items before buying.",
    "phones_accessories": "Double-check your phone model name before ordering.",
    "health_household": "For supplements, check the expiry date and talk to a doctor if unsure.",
    "beauty": "Check the expiry date and do a patch test for new skincare.",
}


def wording(cat, aspect):
    name, tip = CATEGORY_WORDING.get(cat, {}).get(aspect, (aspect, ADVICE.get(aspect, "")))
    return name, tip


GENERAL_ADVICE = [
    "Sort reviews by 'Most recent' to see if quality changed lately.",
    "Prefer reviews marked 'Verified Purchase' with photos.",
    "Check the return policy before buying.",
]

# ---------- tuning knobs ----------
PRIOR_REVIEWS = 5        # a product with few reviews is pulled toward the category average
GOOD_SIMILARITY = 0.25   # average similarity at which we fully trust the neighbours (calibrated on the v2 indexes)
CALIBRATION_SAMPLE = 400 # products used at startup to learn what a "normal" score looks like
HIGH_RISK_PCT = 20       # bottom 20% of the category = High risk
MEDIUM_RISK_PCT = 40     # bottom 20-40% = Medium risk
MIN_NEIGHBOUR_REVIEWS = 3  # products with fewer reviews are too unreliable to use as look-alikes
MIN_GAP = 0.15           # High risk also needs to be at least this many stars below the category average
REVIEW_PRIOR = 20        # real page reviews: our estimate counts like this many extra reviews
BRAND_WEIGHT = 0.3       # how much a known brand's track record moves the estimate
MIN_BRAND_PRODUCTS = 3   # a brand needs at least this many products in the data to count

# First words of titles that are not brands
NOT_BRANDS = {"amazon", "cable", "usb", "women", "womens", "men", "mens", "new", "the", "for", "wireless",
              "bluetooth", "portable", "pack", "set", "kids", "girls", "boys", "unisex", "premium", "dr", "just",
              "under", "tp", "fire", "echo", "kindle", "fruit", "2", "3", "4", "5", "10", "a", "my", "mini"}


def norm_brand(text):
    """'L'Oréal Paris ...' -> 'loreal'. Uses the first word only."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().strip()
    first = re.split(r"[\s,|/()\-]+", t)[0] if t else ""
    b = re.sub(r"[^a-z0-9&]", "", first)
    return "" if b in NOT_BRANDS or len(b) < 2 else b


def neighbour_estimate(sims, ratings, top_n):
    """Similarity-weighted average rating of the top_n closest products, plus how much they disagree."""
    n = min(top_n, len(sims))
    top = np.argsort(sims)[-n:][::-1]
    w = np.clip(sims[top], 1e-6, None)
    r = ratings[top]
    mean = float(np.sum(w * r) / np.sum(w))
    spread = float(np.sqrt(np.sum(w * (r - mean) ** 2) / np.sum(w)))
    return top, mean, spread


def build_index(label, vec, mat, prof):
    ratings = prof["Rating"].astype(float).to_numpy()
    cat_mean = float(ratings.mean())

    # Products with only a handful of reviews have unreliable averages, so blend them with the category average.
    if "review_count" in prof.columns:
        n = prof["review_count"].astype(float).to_numpy()
        adjusted = (n * ratings + PRIOR_REVIEWS * cat_mean) / (n + PRIOR_REVIEWS)
    else:
        adjusted = ratings

    # Calibration: for a sample of real products, find their neighbours and average them the same way
    # a prediction does. The spread of these numbers tells us what is low or high *for this category*.
    rng = np.random.default_rng(0)
    sample = rng.choice(mat.shape[0], size=min(CALIBRATION_SAMPLE, mat.shape[0]), replace=False)
    S = (mat[sample] @ mat.T).toarray()
    S[np.arange(len(sample)), sample] = -1  # ignore the product itself
    usable = (prof["review_count"].to_numpy() >= MIN_NEIGHBOUR_REVIEWS) if "review_count" in prof.columns \
        else np.ones(len(prof), bool)
    S[:, ~usable] = -1
    calib = np.sort([neighbour_estimate(row, adjusted, 8)[1] for row in S])

    # How common each word is in this category (recovered from the vectorizer), used to pick the category.
    n_docs = mat.shape[0]
    df = (1 + n_docs) / np.exp(vec.idf_ - 1) - 1
    word_frac = {w: df[i] / n_docs for w, i in vec.vocabulary_.items() if " " not in w}

    # Brand track record: average (adjusted) rating of each brand's products in this category.
    brands = {}
    if "title" in prof.columns:
        b = prof["title"].fillna("").map(norm_brand)
        for name, grp in pd.Series(adjusted).groupby(b.values):
            if name and len(grp) >= MIN_BRAND_PRODUCTS:
                brands[name] = (len(grp), float(grp.mean()))

    return {
        "label": label,
        "brands": brands,
        "prod_dist": np.sort(adjusted[usable]),
        "word_frac": word_frac,
        "n_docs": n_docs,
        "vec": vec,
        "mat": mat,
        "prof": prof,
        "adjusted": adjusted,
        "mean": cat_mean,
        "usable": usable,
        "calib": calib,
    }


indexes = {}
for key, (label, vec_file, mat_file, prof_file) in CATEGORIES.items():
    if all(os.path.exists(f) for f in (vec_file, mat_file, prof_file)):
        prof = pd.read_csv(prof_file)
        if "Clothing ID" in prof.columns:
            prof = prof.rename(columns={"Clothing ID": "product_id"})
        prof["product_id"] = prof["product_id"].astype(str)
        indexes[key] = build_index(label, joblib.load(vec_file), joblib.load(mat_file), prof)
        ins_path = insights_file(vec_file)
        indexes[key]["insights"] = json.load(open(ins_path, encoding="utf-8")) if os.path.exists(ins_path) else {}
print("Loaded categories:", list(indexes))


class ReviewInput(BaseModel):
    review_text: str = Field(max_length=5000)


class DescriptionInput(BaseModel):
    description: str = Field(min_length=2, max_length=2000)
    top_n: int = 8
    category: str = Field(default="auto", max_length=40)
    page_rating: Optional[float] = None   # rating shown on the shop page, if any
    page_reviews: Optional[int] = None    # how many ratings/reviews the shop page shows
    brand: Optional[str] = None           # brand read from the page, if any


class LinkInput(BaseModel):
    url: str = Field(max_length=2000)
    top_n: int = 8
    category: str = Field(default="auto", max_length=40)


class FeedbackInput(BaseModel):
    description: str = Field(max_length=2000)
    predicted_rating: float
    was_right: bool


@app.get("/")
def home():
    return {"message": "TruPick API is running", "login_required": bool(FIREBASE_PROJECT_ID)}


@app.get("/categories")
def categories():
    return [{"key": "auto", "label": "Auto-detect"}] + [
        {"key": k, "label": v["label"], "products": int(v["n_docs"]), "insights": bool(v["insights"])}
        for k, v in indexes.items()
    ]


@app.post("/predict")
def predict(data: ReviewInput, user: str = Depends(require_user)):
    """Use this when the product already has reviews."""
    text_vector = vectorizer.transform([data.review_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    return {
        "recommended": bool(prediction),
        "confidence": round(float(max(probability)), 2)
    }


def query_words(text):
    return [w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in ENGLISH_STOP_WORDS]


def word_coverage(ix, text):
    """Share of the query's words that this category's data actually knows."""
    words = query_words(text)
    if not words:
        return 0.0
    return sum(w in ix["word_frac"] for w in words) / len(words)


def pct_in(dist, value):
    return round(100 * float(np.searchsorted(dist, value)) / max(len(dist), 1))


def find_brand(ix, brand_text, description):
    """Looks up the brand in this category first, then in all categories."""
    name = norm_brand(brand_text) if brand_text else ""
    if not name:
        # No brand from the page: try the first word of the text, but only if it is a known brand.
        guess = norm_brand(description)
        if not any(guess in v["brands"] for v in indexes.values()):
            return None
        name = guess
    for where in [ix] + [v for v in indexes.values() if v is not ix]:
        if name in where["brands"]:
            count, avg = where["brands"][name]
            return {"key": name, "products": count, "average": round(avg, 2),
                    "percentile": pct_in(where["prod_dist"], avg), "category_label": where["label"],
                    "same_category": where is ix}
    return {"key": name, "products": 0}


def buyers_say(ix, top, sims, cat=None):
    """Combines what buyers of the closest products praised and complained about."""
    prof, ins = ix["prof"], ix.get("insights") or {}
    agg = {}
    for rank, i in enumerate(top):
        info = ins.get(prof.iloc[i]["product_id"])
        if not info:
            continue
        w = max(float(sims[i]), 1e-3)
        for aspect, (pos, neg, qp, qn) in info.items():
            a = agg.setdefault(aspect, {"wp": 0.0, "wn": 0.0, "pos": 0, "neg": 0, "qps": [], "qns": [], "products": 0})
            a["wp"] += w * pos; a["wn"] += w * neg; a["pos"] += pos; a["neg"] += neg; a["products"] += 1
            if qp: a["qps"].append(qp)
            if qn: a["qns"].append(qn)
    love, complain, used = [], [], set()

    def pick(quotes):
        """First quote not already shown for another aspect (so each aspect has its own buyer's words)."""
        for q in quotes:
            if q not in used:
                used.add(q)
                return q
        return ""

    for aspect, a in sorted(agg.items(), key=lambda kv: -(kv[1]["wp"] + kv[1]["wn"])):
        total = a["wp"] + a["wn"]
        if total <= 0:
            continue
        share_pos = a["wp"] / total
        if share_pos >= 0.65 and a["pos"] >= 3:
            love.append({"aspect": aspect, "percent": round(100 * share_pos), "mentions": a["pos"] + a["neg"],
                         "quote": pick(a["qps"]), "score": a["wp"]})
        if share_pos <= 0.65 and a["neg"] >= 2:
            complain.append({"aspect": aspect, "percent": round(100 * (1 - share_pos)), "mentions": a["pos"] + a["neg"],
                             "quote": pick(a["qns"]), "score": a["wn"]})
    love.sort(key=lambda x: -x["score"]); complain.sort(key=lambda x: -x["score"])
    for x in love + complain:
        x.pop("score")
        x["key"] = x["aspect"]
        x["aspect"] = wording(cat, x["key"])[0]
    return {"love": love[:3], "complain": complain[:3], "products_used": sum(1 for i in top if prof.iloc[i]["product_id"] in ins)}


def checklist_for(complain, love=(), cat=None):
    """Things to check before buying, each with the reason it is on the list."""
    items = []
    for c in complain:
        name, tip = wording(cat, c["key"])
        if tip and all(i["aspect"] != name for i in items):
            items.append({"aspect": name, "tip": tip, "kind": "complaint",
                          "why": f"{c['percent']}% of buyer comments about this were complaints"})
    for c in sorted(love, key=lambda x: -x["mentions"]):
        if len(items) >= 3:
            break
        name, tip = wording(cat, c["key"])
        if tip and all(i["aspect"] != name for i in items):
            items.append({"aspect": name, "tip": tip, "kind": "popular",
                          "why": f"One of the things buyers talk about most ({c['mentions']} mentions)"})
    if cat in CATEGORY_GENERAL and len(items) < 4:
        items.append({"aspect": "", "tip": CATEGORY_GENERAL[cat], "kind": "general", "why": ""})
    for tip in GENERAL_ADVICE:
        if len(items) >= 4:
            break
        items.append({"aspect": "", "tip": tip, "kind": "general", "why": ""})
    return items


def alternatives_for(ix, sims, estimate, exclude_text):
    """Better-rated products among the close matches (only where we know real product names)."""
    prof = ix["prof"]
    if "title" not in prof.columns:
        return []
    cand = np.argsort(sims)[-60:][::-1]
    best = float(sims[cand[0]]) if len(cand) else 0
    out = []
    for i in cand:
        if sims[i] < 0.5 * best or sims[i] <= 0:
            break
        title = str(prof.iloc[i]["title"]) if pd.notna(prof.iloc[i]["title"]) else ""
        n = int(prof.iloc[i]["review_count"]) if "review_count" in prof.columns else 0
        if len(title.split()) < 3 or n < 10 or title.lower()[:40] in exclude_text.lower():
            continue
        if ix["adjusted"][i] >= max(estimate + 0.1, ix["mean"]):
            out.append({"title": title, "rating": round(float(prof.iloc[i]["Rating"]), 1), "reviews": n,
                        "similarity": round(float(sims[i]), 3), "_adj": float(ix["adjusted"][i])})
    out.sort(key=lambda x: -x["_adj"])
    for x in out:
        x.pop("_adj")
    return out[:4]


def category_fit(ix, text):
    """How typical the query's words are for this category ("dress" is common in clothing, rare in beauty)."""
    words = query_words(text)
    if not words:
        return -99.0
    floor = 0.5 / ix["n_docs"]
    return float(np.mean([np.log(ix["word_frac"].get(w, 0) + floor) for w in words]))


@app.post("/predict-similar")
def api_predict_similar(data: DescriptionInput, user: str = Depends(require_user)):
    return predict_similar(data)


def predict_similar(data: DescriptionInput):
    """Use this when the product has NO reviews: predicts quality from similar products."""
    if not indexes:
        return {"error": "No category data is loaded on the server."}
    if data.category != "auto" and data.category not in indexes:
        return {"error": "Unknown category."}

    top_n = max(3, min(data.top_n, 20))
    keys = list(indexes) if data.category == "auto" else [data.category]

    # Auto-detect: the category where the query's words are most typical.
    route_text = data.description
    if data.brand:  # a brand name like "boAt" shouldn't decide the category (boat shoes!)
        route_text = re.sub(re.escape(data.brand), " ", route_text, flags=re.I)
    k = max(keys, key=lambda c: category_fit(indexes[c], route_text))
    ix = indexes[k]
    sims = (ix["vec"].transform([data.description]) @ ix["mat"].T).toarray()[0]
    sims[~ix["usable"]] = 0  # skip products with too few reviews
    top, neigh, spread = neighbour_estimate(sims, ix["adjusted"], top_n)

    avg_sim = float(sims[top].mean())
    coverage = word_coverage(ix, data.description)
    match = min(avg_sim / GOOD_SIMILARITY, 1.0)

    # Weak match -> lean toward the category average instead of trusting a few random look-alikes.
    strength = match * (0.5 + 0.5 * coverage)
    estimate = strength * neigh + (1 - strength) * ix["mean"]

    # Confidence of the look-alike estimate: closeness, known words, and whether the look-alikes agree.
    agreement = 1 - min(spread, 1.0)
    conf_score = round(100 * (0.5 * match + 0.25 * coverage + 0.25 * agreement))
    half = max(spread, 0.15) * (1.5 - 0.5 * strength)

    # Brand track record (only when the brand has enough products in our data).
    brand = find_brand(ix, data.brand, data.description)
    if brand and brand["products"] >= MIN_BRAND_PRODUCTS and brand["same_category"]:
        estimate = (1 - BRAND_WEIGHT) * estimate + BRAND_WEIGHT * brand["average"]
        conf_score = min(100, conf_score + 10)

    # Real reviews on the shop page beat any estimate: blend them in by how many there are.
    n = data.page_reviews or 0
    has_reviews = data.page_rating is not None and 1 <= data.page_rating <= 5 and n > 0
    if has_reviews:
        predicted = (n * data.page_rating + REVIEW_PRIOR * estimate) / (n + REVIEW_PRIOR)
        half = max(0.08, half * REVIEW_PRIOR / (n + REVIEW_PRIOR))
        conf_score = max(conf_score, 95 if n >= 200 else 85 if n >= 50 else 70 if n >= 10 else conf_score)
        source = "reviews"
    else:
        predicted = estimate
        source = "estimate"

    low, high = max(1.0, predicted - half), min(5.0, predicted + half)
    confidence = "High" if conf_score >= 75 else "Medium" if conf_score >= 50 else "Low"

    # Where this sits within its own category (0 = worst, 100 = best).
    # Real ratings are compared with real products; estimates with typical look-alike averages.
    dist = ix["prod_dist"] if has_reviews and n >= REVIEW_PRIOR else ix["calib"]
    percentile = pct_in(dist, predicted)

    # Trust score 0-100: the percentile, pulled toward 50 when we are unsure.
    sure = min(conf_score / 75, 1.0)
    trust_score = round(50 + (percentile - 50) * sure)
    if has_reviews:
        # Real ratings: half "how it ranks" and half "how good the stars are" (3.0* = 0, 5.0* = 100),
        # because ranking alone is harsh: our data is US Amazon, where ratings run high.
        stars = min(max((predicted - 3.0) / 2.0, 0), 1) * 100
        trust_score = round(0.5 * percentile + 0.5 * stars)

    if has_reviews:
        # With real ratings, judge the stars themselves.
        risk = "High risk" if predicted < 3.5 else "Medium risk" if predicted < 4.0 else "Low risk"
    elif confidence == "Low":
        risk = "Unclear"
    elif trust_score < HIGH_RISK_PCT and predicted <= ix["mean"] - MIN_GAP:
        risk = "High risk"
    elif trust_score < MEDIUM_RISK_PCT:
        risk = "Medium risk"
    else:
        risk = "Low risk"

    label = ix["label"]
    reasons = []
    if has_reviews:
        reasons.append(f"The shop page shows {data.page_rating:.1f}\u2605 from {n:,} ratings, which counts most.")
    reasons += [
        f"Compared with {len(top)} similar products in {label} (average similarity {avg_sim:.0%}); "
        f"their ratings vary by about \u00b1{spread:.2f} stars.",
        f"{coverage:.0%} of your words appear in the {label} data.",
    ]
    if brand and brand["products"] >= MIN_BRAND_PRODUCTS:
        reasons.append(f"Brand: {brand['products']} products from this brand in {brand['category_label']} "
                       f"average {brand['average']:.2f}\u2605, better than {brand['percentile']}% of products there.")
    elif brand:
        reasons.append("Brand not found in our data, so it isn't used.")
    reasons.append(f"The {label} average is {ix['mean']:.2f}\u2605; this product scores better than "
                   f"{percentile}% of typical {label} products, giving a trust score of {trust_score}/100.")
    if strength < 0.5 and not has_reviews:
        reasons.append("The match is weak, so the score leans toward the middle.")

    prof = ix["prof"]
    said = buyers_say(ix, top, sims, k)
    checklist = checklist_for(said["complain"], said["love"], k)
    alternatives = alternatives_for(ix, sims, predicted, data.description)
    similar = [
        {
            "Clothing ID": str(prof.iloc[i]["product_id"]),
            "title": str(prof.iloc[i]["title"]) if "title" in prof.columns and pd.notna(prof.iloc[i]["title"]) else "",
            "Rating": round(float(prof.iloc[i]["Rating"]), 2),
            "reviews": int(prof.iloc[i]["review_count"]) if "review_count" in prof.columns else None,
            "similarity": round(float(sims[i]), 4),
        }
        for i in top
    ]

    return {
        "trust_score": trust_score,
        "source": source,
        "page_rating": data.page_rating if has_reviews else None,
        "page_reviews": n if has_reviews else None,
        "estimate_rating": round(estimate, 2),
        "brand": brand,
        "predicted_rating": round(predicted, 2),
        "rating_low": round(low, 2),
        "rating_high": round(high, 2),
        "category_average": round(ix["mean"], 2),
        "percentile": percentile,
        "risk": risk,
        "confidence": confidence,
        "confidence_score": conf_score,
        "match_similarity": round(avg_sim, 4),
        "word_coverage": round(coverage, 2),
        "reasons": reasons,
        "similar_products": similar,
        "category": k,
        "category_label": label,
        "buyers_say": said,
        "checklist": checklist,
        "alternatives": alternatives,
    }


def clean_title(t):
    """Drops shop slogans like '- Buy ... Online at Best Price' from page titles."""
    t = re.split(r"\s[-|:–]\s*(Buy|Shop|Online|Amazon|Flipkart|Myntra|Ajio)\b", t, flags=re.I)[0]
    return re.sub(r"\s+", " ", t).strip()


def to_number(x):
    try:
        return float(re.sub(r"[^0-9.]", "", str(x)))
    except ValueError:
        return None


def read_page(html):
    """Returns (product text, page rating, number of ratings, brand) from a product page."""
    soup = BeautifulSoup(html, "html.parser")
    parts, rating, reviews, brand = [], None, None, None

    # 1. Structured product data many shops include for Google (most reliable)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for it in items:
            if isinstance(it, dict) and "Product" in str(it.get("@type", "")):
                parts += [str(it.get("name", "")), str(it.get("description", ""))[:600]]
                b = it.get("brand")
                brand = brand or (b.get("name") if isinstance(b, dict) else b)
                agg = it.get("aggregateRating") or {}
                if isinstance(agg, dict) and agg.get("ratingValue"):
                    rating = to_number(agg.get("ratingValue"))
                    reviews = to_number(agg.get("ratingCount") or agg.get("reviewCount"))

    # 2. Social-share tags, then the normal title / description / main heading
    def meta(**attrs):
        m = soup.find("meta", attrs=attrs)
        return m["content"] if m and m.get("content") else ""

    parts += [meta(property="og:title"), meta(property="og:description"), meta(name="description")]
    if soup.title and soup.title.string:
        parts.append(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        parts.append(h1.get_text(" ", strip=True))

    seen, out = set(), []
    for p in parts:
        p = clean_title(p)
        if len(p) > 3 and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    if rating and not 1 <= rating <= 5:
        rating = None
    return " ".join(out)[:1200], rating, int(reviews) if reviews else None, brand


def safe_public_url(url):
    """Only real public websites: blocks localhost and private network addresses."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
    except Exception:
        return False
    return True


@app.post("/predict-link")
def api_predict_link(data: LinkInput, user: str = Depends(require_user)):
    return predict_link(data)


BLOCKED_HINTS = re.compile(
    r"robot check|captcha|are you a human|not a robot|access denied|request blocked|unusual traffic|"
    r"enable javascript|enable cookies|page not found|something went wrong|sorry!? we|"
    r"online shopping site for|online shopping for|online shopping india|amazon\.in: online shopping|"
    r"buy online at best price in india on flipkart|flipkart\.com: online shopping|shop online for", re.I)
SLUG_SKIP = {"dp", "p", "gp", "product", "products", "buy", "item", "itm", "s", "ref", "www", "html", "amp", "dl", "in", "com"}


def name_from_url(url):
    """Shop links usually contain the product name, e.g. /clayor-ceramic-sugar-container-550-ml/p/itm..."""
    parts = [x for x in re.split(r"[/?#&=]", urlparse(url).path) if x]
    best = ""
    for seg in parts:
        words = [w for w in re.split(r"[-_+]+|%20", seg) if w]
        words = [w for w in words if not re.fullmatch(r"(itm|b0)[0-9a-z]{6,}|[0-9a-f]{10,}|\d{5,}", w, re.I)
                 and w.lower() not in SLUG_SKIP]
        letters = [w for w in words if re.search(r"[a-z]", w, re.I)]
        if len(letters) >= 2 and len(" ".join(words)) > len(best):
            best = " ".join(words)
    return best[:200]


def looks_blocked(text):
    """True when the shop sent a robot check or its homepage instead of the product page."""
    return len(text) < 10 or bool(BLOCKED_HINTS.search(text[:400]))


def predict_link(data: LinkInput):
    """Reads a product page from a link, then predicts quality."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-IN,en;q=0.9",
    }
    url = data.url.strip()
    text, rating, reviews, brand = "", None, None, None
    try:
        for _ in range(5):  # follow redirects ourselves, checking each address
            if not safe_public_url(url):
                return {"error": "That link can't be opened. Please use a normal shop product link."}
            r = requests.get(url, headers=headers, timeout=10, allow_redirects=False, stream=True)
            if r.is_redirect and r.headers.get("location"):
                url = requests.compat.urljoin(url, r.headers["location"])
                continue
            break
        body, size = [], 0
        for chunk in r.iter_content(65536):
            body.append(chunk); size += len(chunk)
            if size > 3_000_000:
                break
        html = b"".join(body).decode(r.encoding or "utf-8", errors="ignore")
        if r.status_code < 400:
            text, rating, reviews, brand = read_page(html)
    except Exception:
        pass  # fall back to the name in the link below

    source = "page"
    slug = name_from_url(url) or name_from_url(data.url)
    if looks_blocked(text):
        # The shop blocked us or sent its homepage: use the product name written in the link instead.
        text, rating, reviews, brand, source = slug, None, None, None, "link"
    if len(text) < 6:
        return {"error": "This shop didn't let us read that page, and the link has no product name in it. "
                         "Please type the product name instead (e.g. 'ceramic sugar container')."}

    result = predict_similar(DescriptionInput(description=text, top_n=data.top_n, category=data.category,
                                              page_rating=rating, page_reviews=reviews,
                                              brand=str(brand) if brand else None))
    if "error" not in result:
        result["page_text"] = text
        result["read_from"] = source
    return result


@app.post("/feedback")
def feedback(data: FeedbackInput, user: str = Depends(require_user)):
    """Stores user feedback so the model can be improved later."""
    new_file = not os.path.exists('feedback.csv')
    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['time', 'description', 'predicted_rating', 'was_right', 'user'])
        w.writerow([datetime.now().isoformat(), data.description, data.predicted_rating, data.was_right, user])
    return {"saved": True}
