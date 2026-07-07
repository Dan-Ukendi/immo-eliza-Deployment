"""
streamlit_app.py

Small web app for non-technical users:
  1. picks which trained model to use (dropdown, shows each model's R²)
  2. fills in a property listing form -- fields are color-coded by how
     much that specific model relies on them (green = most important,
     yellow = neutral, red = least useful), using the model's own
     feature-importance ranking
  3. sends the listing to the deployed FastAPI backend and gets a price
     estimate, the model's precision, a feature-importance chart, and a
     map of the listing's location

The frontend and the API are intentionally separate (per the mission's
architecture) -- this file only talks to the API over HTTP, it never
loads a model directly.
"""

import pandas as pd
import requests
import streamlit as st
import altair as alt

# ====================================================================
#  Config
# ====================================================================
API_URL = "https://immo-eliza-deployment-zla0.onrender.com"

# Render's free tier spins the API down after 15 min of inactivity --
# the first request after a while can take up to ~50s to "wake it up".
REQUEST_TIMEOUT = 60

PROPERTY_SUBTYPES = [
    "apartment", "bungalow", "chalet", "cottage", "duplex",
    "ground-floor", "loft", "mansion", "master-house",
    "mixed-building", "penthouse", "residence", "studio",
    "triplex", "villa",
]
PROVINCES = [
    "Antwerp", "Brussels Capital Region", "East Flanders",
    "Flemish Brabant", "Hainaut", "Limburg", "Liège",
    "Luxembourg", "Namur", "Walloon Brabant", "West Flanders",
]
REGIONS = ["Brussels", "Flanders", "Wallonia"]
KITCHEN_OPTIONS = ["Not equipped", "Partially equipped", "Fully equipped", "Super equipped"]
STATE_OPTIONS = ["To renovate", "Normal", "Fully renovated", "New", "Excellent"]
EPC_OPTIONS = ["G", "F", "E", "E+", "D", "C", "B", "B+", "A", "A+", "A++"]

# Raw model feature names -> readable French labels.
FEATURE_LABELS = {
    "living_area_m2": "Surface habitable",
    "total_area_m2": "Surface totale du terrain",
    "bedrooms": "Chambres",
    "bathrooms": "Salles de bain",
    "building_year": "Année de construction",
    "latitude": "Localisation (nord-sud)",
    "longitude": "Localisation (est-ouest)",
    "nearby_city_price_m2": "Prix moyen de la ville proche",
    "km_from_nearby_city": "Distance à la ville proche",
    "state_encoded": "État du bâtiment",
    "kitchen_equipped_encoded": "Équipement de la cuisine",
    "epc_encoded": "Score PEB",
    "facades": "Nombre de façades",
    "property_subtype": "Sous-type de bien",
    "province": "Province",
    "region": "Région",
    "property_type_encoded": "Type de bien",
    "floor_number": "Étage",
    "parking_count": "Places de parking",
    "garden_area_m2": "Surface du jardin",
    "has_terrace": "Terrasse",
    "furnished": "Meublé",
    "has_garage": "Garage",
    "has_elevator": "Ascenseur",
    "has_garden": "Jardin",
    "is_nearby_city_prestigious": "Ville proche prestigieuse",
}

# Form field key -> the (grouped) feature-importance name it corresponds
# to. Must cover every field in the form below.
FIELD_TO_GROUP = {
    "living_area_m2": "living_area_m2",
    "total_area_m2": "total_area_m2",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "building_year": "building_year",
    "latitude": "latitude",
    "longitude": "longitude",
    "nearby_city": "nearby_city_price_m2",
    "km_from_nearby_city": "km_from_nearby_city",
    "state_of_the_building": "state_encoded",
    "kitchen_equipped": "kitchen_equipped_encoded",
    "epc_score": "epc_encoded",
    "facades": "facades",
    "property_subtype": "property_subtype",
    "province": "province",
    "region": "region",
    "property_type": "property_type_encoded",
    "floor_number": "floor_number",
    "parking_count": "parking_count",
    "garden_area_m2": "garden_area_m2",
    "has_terrace": "has_terrace",
    "furnished": "furnished",
    "has_garage": "has_garage",
    "has_elevator": "has_elevator",
    "has_garden": "has_garden",
    "is_nearby_city_prestigious": "is_nearby_city_prestigious",
}

TIER_EMOJI = {"high": "\U0001F7E2", "mid": "\U0001F7E1", "low": "\U0001F534"}  # 🟢 🟡 🔴


# ====================================================================
#  Visual identity -- custom fonts, palette, and layout accents on top
#  of the base theme in .streamlit/config.toml
# ====================================================================
def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Newsreader', serif;
            color: #1B2224;
            letter-spacing: -0.01em;
        }
        .eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #A3492F;
            border-bottom: 1px solid #D8D3C8;
            padding-bottom: 0.35rem;
            margin: 1.4rem 0 0.9rem 0;
        }
        [data-testid="stForm"] {
            background-color: #F7F6F2;
            border: 1px solid #E2DED2;
            border-radius: 6px;
            padding: 1.6rem 1.8rem;
        }
        .stButton button, [data-testid="stFormSubmitButton"] button {
            background-color: #A3492F;
            color: #F7F6F2;
            border: none;
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
            background-color: #8A3C27;
            color: #FFFFFF;
        }
        .price-plaque {
            background-color: #2B3A42;
            border: 1px solid #B98B34;
            border-radius: 6px;
            padding: 1.6rem 2rem;
            text-align: center;
            margin: 0.6rem 0 1.6rem 0;
        }
        .price-plaque .label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #B98B34;
        }
        .price-plaque .value {
            font-family: 'Newsreader', serif;
            font-weight: 600;
            font-size: 3rem;
            color: #F2EEE6;
            line-height: 1.2;
        }
        .legend-pill {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: #4A4640;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def eyebrow(text: str):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


# ====================================================================
#  Cached calls to the API. Model list/scores and each model's
#  feature-importance ranking are static (don't change between
#  requests), so they're cached to avoid re-fetching on every rerun.
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models():
    response = requests.get(f"{API_URL}/models", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["models"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_feature_importance(model_name: str):
    response = requests.get(
        f"{API_URL}/feature-importance", params={"model": model_name}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["top_features"]


def compute_field_tiers(feature_importance: list | None):
    """
    Turns a model's grouped feature-importance ranking into a per-form-
    field tier ("high"/"mid"/"low", by rank tercile) and a per-field
    importance percentage. Fields whose underlying feature was dropped
    for this model (absent from the ranking) get 0% and land in the
    bottom tier. Returns ({}, {}) if no ranking is available at all
    (e.g. SVM with a non-linear kernel) -- the form then shows plain,
    uncolored labels.
    """
    if not feature_importance:
        return {}, {}

    group_pct = {item["feature"]: item["importance_pct"] for item in feature_importance}
    field_pct = {field: group_pct.get(group, 0.0) for field, group in FIELD_TO_GROUP.items()}

    ranked = sorted(field_pct.items(), key=lambda kv: -kv[1])
    n = len(ranked)
    tiers = {}
    for i, (field, _pct) in enumerate(ranked):
        if i < n / 3:
            tiers[field] = "high"
        elif i < 2 * n / 3:
            tiers[field] = "mid"
        else:
            tiers[field] = "low"
    return tiers, field_pct


def field_label(base_label: str, field_key: str, tiers: dict, field_pct: dict) -> str:
    """Prefixes a form label with a tier emoji + importance % for the
    current model, or returns it unchanged if no ranking is available."""
    tier = tiers.get(field_key)
    if tier is None:
        return base_label
    return f"{TIER_EMOJI[tier]} {base_label} ({field_pct[field_key]:.1f}%)"


# ====================================================================
#  Page
# ====================================================================
st.set_page_config(page_title="Immo Eliza -- Estimation de prix", page_icon="\U0001F3E0", layout="centered")
inject_style()

st.title("Immo Eliza")
st.caption("Estimation de prix immobilier en Belgique -- choisis ton modèle, remplis le bien.")

# ---- Model selection (OUTSIDE the form: changing it must instantly
#      update the color-coding below, forms only rerun on submit) ----
try:
    available_models = fetch_models()
except requests.exceptions.RequestException:
    available_models = []
    st.error("Impossible de récupérer la liste des modèles depuis l'API.")

if available_models:
    available_models = sorted(available_models, key=lambda m: -m["test_scores"]["R2"])
    model_names = [m["name"] for m in available_models]
    model_lookup = {m["name"]: m for m in available_models}

    eyebrow("Modèle de prédiction")
    selected_model = st.selectbox(
        "Choisis un modèle",
        options=model_names,
        format_func=lambda name: f"{model_lookup[name]['display_name']} — R²={model_lookup[name]['test_scores']['R2']:.2f}",
        key="selected_model",
    )

    scores = model_lookup[selected_model]["test_scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{scores['R2']:.2f}")
    c2.metric("RMSE", f"{scores['RMSE']:,.0f} EUR".replace(",", " "))
    c3.metric("MAE", f"{scores['MAE']:,.0f} EUR".replace(",", " "))

    try:
        current_importance = fetch_feature_importance(selected_model)
    except requests.exceptions.RequestException:
        current_importance = None

    tiers, field_pct = compute_field_tiers(current_importance)

    if tiers:
        st.markdown(
            '<span class="legend-pill">🟢 très influent &nbsp;&nbsp; 🟡 neutre &nbsp;&nbsp; 🔴 peu utile pour ce modèle</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            f"Classement d'importance non disponible pour {model_lookup[selected_model]['display_name']} "
            "(ce modèle n'expose pas de coefficient/importance par variable)."
        )
else:
    selected_model = None
    tiers, field_pct = {}, {}


def flabel(base: str, key: str) -> str:
    return field_label(base, key, tiers, field_pct)


with st.form("listing_form"):
    eyebrow("Caractéristiques principales")
    col1, col2, col3 = st.columns(3)
    with col1:
        living_area_m2 = st.number_input(flabel("Surface habitable (m²)", "living_area_m2"), min_value=0, value=100, key="living_area_m2")
        bedrooms = st.number_input(flabel("Chambres", "bedrooms"), min_value=0, value=2, key="bedrooms")
    with col2:
        bathrooms = st.number_input(flabel("Salles de bain", "bathrooms"), min_value=0, value=1, key="bathrooms")
        total_area_m2 = st.number_input(flabel("Surface totale du terrain (m²)", "total_area_m2"), min_value=0, value=100, key="total_area_m2")
    with col3:
        building_year = st.number_input(flabel("Année de construction", "building_year"), min_value=1800, max_value=2026, value=1975, key="building_year")
        floor_number = st.number_input(flabel("Étage", "floor_number"), min_value=0, value=0, key="floor_number")

    eyebrow("Type de bien")
    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox(flabel("Type", "property_type"), ["Apartment", "House"], key="property_type")
    with col2:
        property_subtype = st.selectbox(flabel("Sous-type", "property_subtype"), PROPERTY_SUBTYPES, key="property_subtype")

    eyebrow("Localisation")
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox(flabel("Région", "region"), REGIONS, key="region")
    with col2:
        province = st.selectbox(flabel("Province", "province"), PROVINCES, key="province")
    with col3:
        nearby_city = st.text_input(flabel("Ville la plus proche", "nearby_city"), value="Bruxelles", key="nearby_city")

    col1, col2, col3 = st.columns(3)
    with col1:
        latitude = st.number_input(flabel("Latitude", "latitude"), value=50.85, format="%.5f", key="latitude")
    with col2:
        longitude = st.number_input(flabel("Longitude", "longitude"), value=4.35, format="%.5f", key="longitude")
    with col3:
        km_from_nearby_city = st.number_input(flabel("Distance à la ville (km)", "km_from_nearby_city"), min_value=0.0, value=2.0, key="km_from_nearby_city")

    is_nearby_city_prestigious = st.checkbox(flabel("La ville proche est considérée prestigieuse", "is_nearby_city_prestigious"), key="is_nearby_city_prestigious")

    eyebrow("Confort & état")
    col1, col2, col3 = st.columns(3)
    with col1:
        kitchen_equipped = st.selectbox(flabel("Cuisine", "kitchen_equipped"), KITCHEN_OPTIONS, index=2, key="kitchen_equipped")
    with col2:
        state_of_the_building = st.selectbox(flabel("État du bâtiment", "state_of_the_building"), STATE_OPTIONS, index=1, key="state_of_the_building")
    with col3:
        epc_score = st.selectbox(flabel("Score PEB", "epc_score"), EPC_OPTIONS, index=5, key="epc_score")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        furnished = st.checkbox(flabel("Meublé", "furnished"), key="furnished")
    with col2:
        has_elevator = st.checkbox(flabel("Ascenseur", "has_elevator"), key="has_elevator")
    with col3:
        has_terrace = st.checkbox(flabel("Terrasse", "has_terrace"), key="has_terrace")
    with col4:
        has_garden = st.checkbox(flabel("Jardin", "has_garden"), key="has_garden")

    garden_area_m2 = st.number_input(flabel("Surface du jardin (m²)", "garden_area_m2"), min_value=0, value=0, disabled=not has_garden, key="garden_area_m2")

    eyebrow("Extérieur")
    col1, col2 = st.columns(2)
    with col1:
        has_garage = st.checkbox(flabel("Garage", "has_garage"), key="has_garage")
    with col2:
        facades = st.number_input(flabel("Nombre de façades", "facades"), min_value=1, max_value=4, value=2, key="facades")
    parking_count = st.number_input(flabel("Places de parking", "parking_count"), min_value=0, value=1 if has_garage else 0, key="parking_count")

    submitted = st.form_submit_button("Estimer le prix", width="stretch", disabled=selected_model is None)


# ====================================================================
#  Submission -> call the API
# ====================================================================
if submitted and selected_model is not None:
    payload = {
        "model": selected_model,
        "data": {
            "living_area_m2": living_area_m2,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "latitude": latitude,
            "longitude": longitude,
            "building_year": building_year,
            "property_type": property_type,
            "property_subtype": property_subtype,
            "province": province,
            "region": region,
            "kitchen_equipped": kitchen_equipped,
            "state_of_the_building": state_of_the_building,
            "epc_score": epc_score,
            "nearby_city": nearby_city,
            "furnished": int(furnished),
            "has_garage": int(has_garage),
            "parking_count": parking_count,
            "has_elevator": int(has_elevator),
            "facades": facades,
            "has_garden": int(has_garden),
            "garden_area_m2": garden_area_m2,
            "has_terrace": int(has_terrace),
            "total_area_m2": total_area_m2,
            "km_from_nearby_city": km_from_nearby_city,
            "is_nearby_city_prestigious": int(is_nearby_city_prestigious),
            "floor_number": floor_number,
        },
    }

    response = None
    with st.spinner("Estimation en cours... (le serveur peut mettre jusqu'à 50s à se réveiller sur le plan gratuit)"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            st.error("Le serveur met trop de temps à répondre. Réessaie dans quelques secondes.")
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifie qu'elle est bien déployée et en ligne.")

    if response is not None:
        if response.status_code == 200:
            price = response.json().get("prediction")
            if price is None:
                st.warning("L'API n'a pas pu produire d'estimation pour ce bien.")
            else:
                formatted_price = f"{price:,.0f}".replace(",", " ")
                st.markdown(
                    f"""
                    <div class="price-plaque">
                        <div class="label">Prix estimé -- {model_lookup[selected_model]['display_name']}</div>
                        <div class="value">{formatted_price} EUR</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                eyebrow("Ce qui influence le plus le prix pour ce modèle")
                if current_importance:
                    chart_df = pd.DataFrame(current_importance[:8])
                    chart_df["label"] = chart_df["feature"].map(
                        lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").capitalize())
                    )
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar(color="#5B7160")
                        .encode(
                            x=alt.X("importance_pct:Q", title="Importance (%)"),
                            y=alt.Y("label:N", sort="-x", title=None),
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.caption("Détail des features indisponible pour ce modèle.")

                eyebrow("Localisation du bien")
                st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), size=200, color="#A3492F")
        else:
            st.error(f"Erreur API (status {response.status_code}). Réessaie ou vérifie les champs saisis.")
