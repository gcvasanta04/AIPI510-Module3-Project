"""Streamlit front end. Calls the deployed API; it never loads the model itself.

Local:  API_URL=http://localhost:8000 streamlit run frontend/app.py
Cloud:  set API_URL in the app's Secrets on Streamlit Community Cloud.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="DC Airbnb Price Estimator", page_icon="🏠", layout="centered")


def api_url() -> str:
    try:
        url = st.secrets["API_URL"]
    except Exception:  # no secrets.toml locally -> fall back to env var
        url = os.environ.get("API_URL", "http://localhost:8000")
    return url.rstrip("/")


API = api_url()


@st.cache_data(ttl=3600, show_spinner="Waking up the API...")
def get_metadata() -> dict:
    r = requests.get(f"{API}/metadata", timeout=30)  # first call may be a Lambda cold start
    r.raise_for_status()
    return r.json()


def predict(payload: dict) -> dict:
    r = requests.post(f"{API}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


st.title("🏠 DC Airbnb Nightly Price Estimator")
st.caption("Describe a listing and a model trained on Inside Airbnb data for Washington, DC "
           "estimates its nightly price. This is a class project, not pricing advice.")

try:
    meta = get_metadata()
except requests.RequestException as e:
    st.error(f"Couldn't reach the API at {API}. ({e})")
    st.stop()

cats, ranges = meta["categories"], meta["numeric_ranges"]

with st.form("listing"):
    c1, c2 = st.columns(2)
    neighbourhood = c1.selectbox("Neighbourhood", cats["neighbourhood"])
    room_type = c2.selectbox("Room type", cats["room_type"])

    c1, c2, c3, c4 = st.columns(4)
    accommodates = c1.number_input("Guests", 1, 16, 2)
    bedrooms = c2.number_input("Bedrooms", 0, 10, 1)
    beds = c3.number_input("Beds", 0, 20, 1)
    bathrooms = c4.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)

    c1, c2 = st.columns(2)
    minimum_nights = c1.number_input("Minimum nights", 1, 365, 2)
    number_of_reviews = c2.number_input("Number of reviews", 0, 2000, 10)
    has_rating = st.checkbox("Listing has a rating", value=True)
    rating = st.slider("Average rating", 1.0, 5.0, 4.8, 0.05, disabled=not has_rating)

    c1, c2, c3 = st.columns(3)
    shared_bath = c1.checkbox("Shared bathroom")
    superhost = c2.checkbox("Superhost")
    instant = c3.checkbox("Instant book")

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

payload = {
    "neighbourhood": neighbourhood, "room_type": room_type,
    "accommodates": int(accommodates), "bedrooms": float(bedrooms), "beds": float(beds),
    "bathrooms": float(bathrooms), "shared_bath": shared_bath,
    "minimum_nights": int(minimum_nights), "number_of_reviews": int(number_of_reviews),
    "review_scores_rating": float(rating) if has_rating else None,
    "host_is_superhost": superhost, "instant_bookable": instant,
}

if submitted:
    try:
        res = predict(payload)
    except requests.RequestException as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.metric("Estimated nightly price", f"${res['predicted_price']:,.0f}")
    st.write(f"Typical range: **\\${res['range_low']:,.0f} – \\${res['range_high']:,.0f}** "
             f"(about {res['range_coverage']:.0%} of similar listings fall in this range "
             f"on held-out data)")

    st.subheader("What if you hosted more guests?")
    rows = []
    for g in range(1, 9):
        r = predict(payload | {"accommodates": g})
        rows.append({"Guests": g, "Estimated price ($)": r["predicted_price"]})
    st.bar_chart(pd.DataFrame(rows).set_index("Guests"))

    with st.expander("Request / response JSON"):
        st.json({"request": payload, "response": res})

with st.sidebar:
    st.header("Model")
    m = meta["metrics"]
    st.write(f"Type: `{meta['model_type']}`")
    st.write(f"Trained: {meta['trained_at'][:10]}")
    st.metric("Test MAE", f"${m['test_mae']:,.0f}",
              delta=f"{m['test_mae'] - m['baseline_val_mae']:,.0f} vs. median baseline",
              delta_color="inverse")
    st.metric("Test R²", f"{m['test_r2']:.2f}")
    st.caption(f"API: {API}")
