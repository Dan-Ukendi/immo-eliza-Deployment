"""
streamlit_app.py

Small web app for non-technical users: fills in a property listing form
and sends it to the deployed FastAPI backend to get a price estimate.

The frontend and the API are intentionally separate (per the mission's
architecture) -- this file only talks to the API over HTTP, it never
loads the model directly.
"""

import requests
import streamlit as st

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


# ====================================================================
#  Page
# ====================================================================
st.set_page_config(page_title="Immo Eliza -- Price Estimator", page_icon="\U0001F3E0")
st.title("\U0001F3E0 Immo Eliza -- Estimation de prix")
st.caption("Remplis les caractéristiques du bien, le modèle Random Forest estime son prix.")

with st.form("listing_form"):
    st.subheader("Caractéristiques principales")
    col1, col2, col3 = st.columns(3)
    with col1:
        living_area_m2 = st.number_input("Surface habitable (m²)", min_value=0, value=100)
        bedrooms = st.number_input("Chambres", min_value=0, value=2)
    with col2:
        bathrooms = st.number_input("Salles de bain", min_value=0, value=1)
        total_area_m2 = st.number_input("Surface totale du terrain (m²)", min_value=0, value=100)
    with col3:
        building_year = st.number_input("Année de construction", min_value=1800, max_value=2026, value=1975)
        floor_number = st.number_input("Étage", min_value=0, value=0)

    st.subheader("Type de bien")
    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox("Type", ["Apartment", "House"])
    with col2:
        property_subtype = st.selectbox("Sous-type", PROPERTY_SUBTYPES)

    st.subheader("Localisation")
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Région", REGIONS)
    with col2:
        province = st.selectbox("Province", PROVINCES)
    with col3:
        nearby_city = st.text_input("Ville la plus proche", value="Bruxelles")

    col1, col2, col3 = st.columns(3)
    with col1:
        latitude = st.number_input("Latitude", value=50.85, format="%.5f")
    with col2:
        longitude = st.number_input("Longitude", value=4.35, format="%.5f")
    with col3:
        km_from_nearby_city = st.number_input("Distance à la ville (km)", min_value=0.0, value=2.0)

    is_nearby_city_prestigious = st.checkbox("La ville proche est considérée prestigieuse")

    st.subheader("Confort & état")
    col1, col2, col3 = st.columns(3)
    with col1:
        kitchen_equipped = st.selectbox("Cuisine", KITCHEN_OPTIONS, index=2)
    with col2:
        state_of_the_building = st.selectbox("État du bâtiment", STATE_OPTIONS, index=1)
    with col3:
        epc_score = st.selectbox("Score PEB", EPC_OPTIONS, index=5)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        furnished = st.checkbox("Meublé")
    with col2:
        has_elevator = st.checkbox("Ascenseur")
    with col3:
        has_terrace = st.checkbox("Terrasse")
    with col4:
        has_garden = st.checkbox("Jardin")

    garden_area_m2 = st.number_input("Surface du jardin (m²)", min_value=0, value=0, disabled=not has_garden)

    st.subheader("Extérieur")
    col1, col2 = st.columns(2)
    with col1:
        has_garage = st.checkbox("Garage")
    with col2:
        facades = st.number_input("Nombre de façades", min_value=1, max_value=4, value=2)
    parking_count = st.number_input("Places de parking", min_value=0, value=1 if has_garage else 0)

    submitted = st.form_submit_button("Estimer le prix", use_container_width=True)


# ====================================================================
#  Submission -> call the API
# ====================================================================
if submitted:
    payload = {
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
        }
    }

    with st.spinner("Estimation en cours... (le serveur peut mettre jusqu'à 50s à se réveiller sur le plan gratuit)"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            st.error("Le serveur met trop de temps à répondre. Réessaie dans quelques secondes.")
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifie qu'elle est bien déployée et en ligne.")
        else:
            if response.status_code == 200:
                body = response.json()
                price = body.get("prediction")
                if price is not None:
                    st.success("Estimation terminée !")
                    st.metric("Prix estimé", f"{price:,.0f} EUR")
                else:
                    st.warning("L'API n'a pas pu produire d'estimation pour ce bien.")
            else:
                st.error(f"Erreur API (status {response.status_code}). Réessaie ou vérifie les champs saisis.")
