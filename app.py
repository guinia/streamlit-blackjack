"""
Streamlit app: Blackjack assistant with a pre-trained model.
"""

import os
from typing import List

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from ml_components import (
    RANKS,
    VALUES,
    hand_value,
    BlackjackFeatureExtractor,  # noqa: F401
    DropColumns,  # noqa: F401
)

DEALER_RANKS = [rank for rank in RANKS if rank not in {"J", "Q", "K"}]


MODEL_PATH = "models/blackjack_action_model.joblib"


st.set_page_config(page_title="Blackjack ML", page_icon=":spades:", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


class HandState:
    def __init__(self, cards: List[str], step: int = 1):
        self.cards = [c.upper() for c in cards if c]
        self.step = step
        self.value = hand_value(",".join(self.cards)) if self.cards else 0

    def can_split(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def can_double(self) -> bool:
        return self.step == 1


def basic_strategy(hand: HandState, dealer_up: str) -> str:
    """Simplified basic strategy covering splits, soft totals, and hard totals."""
    dealer_val = VALUES.get(dealer_up, 0)
    if dealer_val == 0:
        return "hit"

    if hand.can_split():
        rank = hand.cards[0]
        if rank == "A":
            return "split"
        if rank == "10":
            return "stand"
        if rank == "9":
            return "split" if (2 <= dealer_val <= 9 and dealer_val != 7) else "stand"
        if rank == "8":
            return "split"
        if rank == "7":
            return "split" if 2 <= dealer_val <= 7 else "hit"
        if rank == "6":
            return "split" if 2 <= dealer_val <= 6 else "hit"
        if rank == "5":
            return "double" if (2 <= dealer_val <= 9 and hand.can_double()) else "hit"
        if rank == "4":
            return "split" if 5 <= dealer_val <= 6 else "hit"
        if rank in ("3", "2"):
            return "split" if 2 <= dealer_val <= 7 else "hit"

    total = 0
    aces = 0
    for card in hand.cards:
        if card == "A":
            total += 11
            aces += 1
        else:
            total += VALUES.get(card, 0)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    is_soft = aces > 0 and total == hand.value and hand.value <= 21

    if is_soft:
        if hand.value == 20:
            return "stand"
        if hand.value == 19:
            return "double" if dealer_val == 6 and hand.can_double() else "stand"
        if hand.value == 18:
            if 2 <= dealer_val <= 6 and hand.can_double():
                return "double"
            if 9 <= dealer_val <= 11:
                return "hit"
            return "stand"
        if hand.value == 17:
            return "double" if 3 <= dealer_val <= 6 and hand.can_double() else "hit"
        if hand.value in (15, 16):
            return "double" if 4 <= dealer_val <= 6 and hand.can_double() else "hit"
        if hand.value in (13, 14):
            return "double" if 5 <= dealer_val <= 6 and hand.can_double() else "hit"

    if hand.value >= 17:
        return "stand"
    if 13 <= hand.value <= 16:
        return "stand" if dealer_val < 7 else "hit"
    if hand.value == 12:
        return "stand" if 4 <= dealer_val <= 6 else "hit"
    if hand.value == 11:
        return "double" if hand.can_double() else "hit"
    if hand.value == 10:
        return "double" if dealer_val <= 9 and hand.can_double() else "hit"
    if hand.value == 9:
        return "double" if 3 <= dealer_val <= 6 and hand.can_double() else "hit"
    return "hit"


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


st.title("Blackjack ML - Jugar con modelo pre-entrenado")
st.caption("Esta version de la app carga automaticamente `models/blackjack_action_model.joblib` y permite jugar directamente.")

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"No se encontro el archivo `{MODEL_PATH}`. Coloca el modelo entrenado en esa ruta y recarga la app.")
    st.stop()

st.success("Modelo cargado correctamente. Listo para jugar!")

tab_play, tab_heatmap = st.tabs(["Jugar", "Heatmap"])

with tab_play:
    st.subheader("Ingresa tu mano")
    col_cards, col_dealer, col_step = st.columns([2, 1, 1])
    with col_cards:
        player_cards_input = st.text_input("Tus cartas (ej: `A,7` o `10,6,2`)", value="A,7")
    with col_dealer:
        dealer_up = st.selectbox("Carta visible del dealer", DEALER_RANKS, index=DEALER_RANKS.index("6"))
    with col_step:
        step = st.number_input("Paso (1 permite `double`)", min_value=1, value=1, step=1)

    cleaned_cards = [c.strip().upper() for c in player_cards_input.split(",") if c.strip()]
    player_cards_csv = ",".join(cleaned_cards)

    invalid_faces = [c for c in cleaned_cards if c in {"J", "Q", "K"}]

    if invalid_faces:
        st.error("Ingresa cartas de figura como 10 antes de consultar al modelo.")
    elif not cleaned_cards:
        st.info("Ingresa al menos una carta para consultar al modelo.")
    else:
        base_row = {
            "player_cards": player_cards_csv,
            "player_cards_step": player_cards_csv,
            "dealer_cards": dealer_up,
            "dealer_visible_card": dealer_up,
            "actions": "",
            "step": step,
            "player_total": hand_value(player_cards_csv),
            "cards_remaining": 0,
            "decks_remaining": 0,
            "running_count_end": 0,
            "true_count_end": 0,
            "true_count_prev_round": 0,
            "bet_amount": 0,
            "final_result": "",
        }

        features = pd.DataFrame([base_row])

        try:
            prediction = model.predict(features)[0]
            st.markdown(f"### El modelo sugiere: **{prediction}**")
        except Exception as exc:
            st.error(f"No se pudo obtener una prediccion del modelo: {exc}")
        else:
            hand = HandState(cleaned_cards, step=step)
            basic_move = basic_strategy(hand, dealer_up)
            st.caption(f"Estrategia basica recomienda: {basic_move}")

            with st.expander("Ver datos utilizados para la prediccion"):
                st.dataframe(features, use_container_width=True)

with tab_heatmap:
    st.subheader("Heatmap de acciones recomendadas")
    st.caption("El modelo se consulta para cada combinacion de total del jugador y carta visible del dealer.")

    total_range = st.slider("Rango de totales del jugador", 4, 21, (5, 20))
    totals = list(range(total_range[0], total_range[1] + 1))

    grid = pd.DataFrame(
        [(total, dealer) for total in totals for dealer in DEALER_RANKS],
        columns=["player_total", "dealer_visible_card"],
    )

    try:
        grid["pred_action"] = model.predict(grid[["player_total", "dealer_visible_card"]])
    except Exception as exc:
        st.error(f"No se pudo generar el heatmap: {exc}")
    else:
        action_palette = {"hit": "#2196F3", "stand": "#4CAF50"}
        chart = (
            alt.Chart(grid)
            .mark_rect()
            .encode(
                x=alt.X("dealer_visible_card:N", title="Carta visible del dealer", sort=DEALER_RANKS),
                y=alt.Y("player_total:O", title="Suma de cartas del jugador", sort=totals),
                color=alt.Color(
                    "pred_action:N",
                    title="Accion recomendada",
                    scale=alt.Scale(
                        domain=list(action_palette.keys()),
                        range=list(action_palette.values()),
                    ),
                ),
                tooltip=["player_total", "dealer_visible_card", "pred_action"],
            )
            .properties(height=500)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Ver datos del heatmap"):
            st.dataframe(grid, use_container_width=True)
