"""
Streamlit app: asistente de Blackjack con modelo pre-entrenado.
"""

import itertools
import os
import random
from typing import List

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from ml_components import (
    RANKS,
    VALUES,
    BlackjackFeatureExtractor,  # noqa: F401
    DropColumns,  # noqa: F401
    hand_value,
    parse_cards,
)


MODEL_PATH = "models/blackjack_action_model.joblib"
DEALER_RANKS = [rank for rank in RANKS if rank not in {"J", "Q", "K"}]


st.set_page_config(page_title="Blackjack ML", page_icon=":spades:", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; }
    .app-panel { background:#141823; border:1px solid #242b3a; border-radius:18px; padding:1.2rem; }
    .cards-row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin: 8px 0; }
    .cardface {
      position: relative;
      width: 64px; height: 88px;
      background: linear-gradient(180deg, #ffffff, #f5f5f5);
      border-radius: 12px;
      border: 1px solid #d5d5d5;
      box-shadow: 0 4px 12px rgba(0,0,0,.28);
      display:flex;
      justify-content:center;
      align-items:center;
    }
    .cardface .corner {
      position: absolute;
      font-weight: 700;
      font-size: 15px;
      line-height: 1;
    }
    .cardface .tl { top: 6px; left: 8px; text-align: left; }
    .cardface .br { bottom: 6px; right: 8px; text-align: right; transform: rotate(180deg); }
    .cardface .suit { font-size: 18px; display:block; }
    .cardface.red { color:#b3122f; }
    .cardface.black { color:#111; }
    .cardback {
      width: 64px; height: 88px;
      border-radius: 12px;
      border: 1px solid #253264;
      background: repeating-linear-gradient(45deg, #1b2340 0 7px, #2b3a74 7px 14px);
      box-shadow: 0 4px 12px rgba(0,0,0,.28);
    }
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


SUITS = ["♠", "♥", "♦", "♣"]
DISPLAY_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
FACE_TO_MODEL = {"J": "10", "Q": "10", "K": "10"}


def build_shoe(num_decks: int) -> List[dict]:
    cards = []
    for _ in range(num_decks):
        for rank in DISPLAY_RANKS:
            for suit in SUITS:
                cards.append(
                    {
                        "display_rank": rank,
                        "model_rank": FACE_TO_MODEL.get(rank, rank),
                        "suit": suit,
                    }
                )
    random.shuffle(cards)
    return cards


def ensure_shoe():
    if not st.session_state.shoe:
        st.session_state.shoe = build_shoe(st.session_state.num_decks)


def draw_card() -> dict:
    ensure_shoe()
    return st.session_state.shoe.pop()


def append_card(target: str, card: dict) -> None:
    value = card["model_rank"]
    display = f"{card['display_rank']}{card['suit']}"
    st.session_state[f"{target}_cards_values"].append(value)
    st.session_state[f"{target}_cards_display"].append(display)


def split_rank_suit(card_str: str) -> tuple[str, str]:
    digits = "".join(ch for ch in card_str if ch.isdigit())
    if digits:
        rank = digits
        suit = card_str[len(digits) :]
    else:
        rank = card_str[:1]
        suit = card_str[1:]
    suit = suit or "♠"
    return rank, suit


def render_card_html(card_str: str, hidden: bool = False) -> str:
    if hidden:
        return '<div class="cardback"></div>'
    rank, suit = split_rank_suit(card_str)
    color_class = "red" if suit in ("♥", "♦") else "black"
    return (
        f'<div class="cardface {color_class}">'
        f'<div class="corner tl">{rank}<span class="suit">{suit}</span></div>'
        f'<div class="corner br">{rank}<span class="suit">{suit}</span></div>'
        "</div>"
    )


def render_cards_html(cards: List[str], hide_second: bool = False) -> str:
    if not cards:
        return '<div class="cardback"></div>'
    html = []
    for idx, card in enumerate(cards):
        html.append(render_card_html(card, hidden=(hide_second and idx == 1)))
    return "".join(html)


def cards_csv(values: List[str]) -> str:
    return ",".join(values)


def current_player_total() -> int:
    return hand_value(cards_csv(st.session_state.player_cards_values))


def current_dealer_total() -> int:
    return hand_value(cards_csv(st.session_state.dealer_cards_values))


def dealer_up_value() -> str:
    return st.session_state.dealer_cards_values[0] if st.session_state.dealer_cards_values else ""


def reset_round_state():
    st.session_state.player_cards_values = []
    st.session_state.player_cards_display = []
    st.session_state.dealer_cards_values = []
    st.session_state.dealer_cards_display = []
    st.session_state.step = 1
    st.session_state.round_over = True
    st.session_state.dealer_hidden = True
    st.session_state.last_rec = None
    st.session_state.last_prob = None
    st.session_state.last_basic = None
    st.session_state.result_text = ""


def ensure_state_defaults():
    if "num_decks" not in st.session_state:
        st.session_state.num_decks = 4
    if "shoe" not in st.session_state:
        st.session_state.shoe = build_shoe(st.session_state.num_decks)
    if "player_cards_values" not in st.session_state:
        st.session_state.player_cards_values = []
    if "player_cards_display" not in st.session_state:
        st.session_state.player_cards_display = []
    if "dealer_cards_values" not in st.session_state:
        st.session_state.dealer_cards_values = []
    if "dealer_cards_display" not in st.session_state:
        st.session_state.dealer_cards_display = []
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "round_over" not in st.session_state:
        st.session_state.round_over = True
    if "dealer_hidden" not in st.session_state:
        st.session_state.dealer_hidden = True
    if "last_rec" not in st.session_state:
        st.session_state.last_rec = None
    if "last_prob" not in st.session_state:
        st.session_state.last_prob = None
    if "last_basic" not in st.session_state:
        st.session_state.last_basic = None
    if "result_text" not in st.session_state:
        st.session_state.result_text = ""


def start_new_round():
    reset_round_state()
    ensure_shoe()
    for _ in range(2):
        append_card("player", draw_card())
    append_card("dealer", draw_card())
    append_card("dealer", draw_card())
    st.session_state.dealer_hidden = True
    st.session_state.round_over = False
    st.session_state.step = 1
    update_recommendations()


def player_hit_action():
    if st.session_state.round_over:
        return
    append_card("player", draw_card())
    st.session_state.step += 1
    if current_player_total() > 21:
        st.session_state.round_over = True
        st.session_state.dealer_hidden = False
        st.session_state.result_text = "Te pasaste. Dealer gana."
    update_recommendations()


def dealer_playout():
    while current_dealer_total() < 17:
        append_card("dealer", draw_card())


def evaluate_round_outcome():
    player_total = current_player_total()
    dealer_total = current_dealer_total()
    if player_total > 21:
        st.session_state.result_text = "Te pasaste. Dealer gana."
    else:
        if dealer_total > 21:
            st.session_state.result_text = "Dealer se pasa. ¡Ganas!"
        elif dealer_total > player_total:
            st.session_state.result_text = f"Dealer {dealer_total} vs Jugador {player_total}. Pierdes."
        elif dealer_total < player_total:
            st.session_state.result_text = f"Jugador {player_total} vs Dealer {dealer_total}. ¡Ganas!"
        else:
            st.session_state.result_text = f"Empate: ambos con {player_total}."


def player_stand_action():
    if st.session_state.round_over:
        return
    st.session_state.round_over = True
    st.session_state.dealer_hidden = False
    dealer_playout()
    evaluate_round_outcome()
    update_recommendations()


def update_recommendations():
    values = st.session_state.player_cards_values
    dealer_vals = st.session_state.dealer_cards_values
    if not values or not dealer_vals:
        st.session_state.last_rec = None
        st.session_state.last_prob = None
        st.session_state.last_basic = None
        return

    player_total = current_player_total()
    dealer_up = dealer_up_value()

    features = pd.DataFrame(
        {
            "player_total": [player_total],
            "dealer_visible_card": [dealer_up],
        }
    )

    try:
        pred = model.predict(features)[0]
        st.session_state.last_rec = pred
        if hasattr(model, "predict_proba"):
            proba = max(model.predict_proba(features)[0])
            st.session_state.last_prob = float(proba)
        else:
            st.session_state.last_prob = None
    except Exception:
        st.session_state.last_rec = None
        st.session_state.last_prob = None

    hand = HandState(values, step=st.session_state.step)
    st.session_state.last_basic = basic_strategy(hand, dealer_up) if dealer_up else None

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


st.title("Blackjack ML - Jugar con modelo pre-entrenado")
st.caption("La app carga el modelo desde `models/blackjack_action_model.joblib` y permite explorar sugerencias.")

model = load_model(MODEL_PATH)
if model is None:
    st.error(f"No se encontro el archivo `{MODEL_PATH}`. Coloca el modelo entrenado en esa ruta y recarga la app.")
    st.stop()

st.success("Modelo cargado correctamente. Listo para jugar!")

ensure_state_defaults()
if (
    st.session_state.player_cards_values
    and st.session_state.dealer_cards_values
    and st.session_state.last_rec is None
):
    update_recommendations()

tab_play, tab_heatmap = st.tabs(["Jugar", "Heatmap"])

with tab_play:
    st.subheader("Mesa de juego")
    deck_options = [1, 2, 4, 6, 8]
    current_idx = deck_options.index(st.session_state.num_decks)
    selected_decks = st.selectbox("Número de mazos", deck_options, index=current_idx)
    if selected_decks != st.session_state.num_decks:
        st.session_state.num_decks = selected_decks
        st.session_state.shoe = build_shoe(selected_decks)
        reset_round_state()

    control_cols = st.columns(4)
    with control_cols[0]:
        if st.button("Nueva mano", use_container_width=True):
            start_new_round()
    with control_cols[1]:
        if st.button("Pedir carta (Hit)", use_container_width=True, disabled=st.session_state.round_over):
            player_hit_action()
    with control_cols[2]:
        if st.button("Plantarse (Stand)", use_container_width=True, disabled=st.session_state.round_over):
            player_stand_action()
    with control_cols[3]:
        if st.button("Reset zapato", use_container_width=True):
            st.session_state.shoe = build_shoe(st.session_state.num_decks)
            reset_round_state()

    dealer_html = render_cards_html(
        st.session_state.dealer_cards_display,
        hide_second=st.session_state.dealer_hidden,
    )
    player_html = render_cards_html(st.session_state.player_cards_display)
    panel_html = f"""
    <div class="app-panel">
        <div><strong>Dealer</strong></div>
        <div class="cards-row">{dealer_html}</div>
        <div style="margin-top:12px;"><strong>Jugador</strong></div>
        <div class="cards-row">{player_html}</div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)

    player_total = current_player_total() if st.session_state.player_cards_values else "-"
    dealer_up = dealer_up_value() or "-"
    dealer_total_display = (
        current_dealer_total()
        if st.session_state.dealer_cards_values and not st.session_state.dealer_hidden
        else "?"
    )

    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(f"**Total jugador:** {player_total}")
    with info_cols[1]:
        st.markdown(f"**Carta visible dealer:** {dealer_up}")
    with info_cols[2]:
        st.markdown(f"**Total dealer:** {dealer_total_display}")

    rec_cols = st.columns(2)
    with rec_cols[0]:
        if st.session_state.last_rec:
            prob_text = (
                f" ({st.session_state.last_prob:.0%})"
                if st.session_state.last_prob is not None
                else ""
            )
            st.success(f"Modelo sugiere: {st.session_state.last_rec}{prob_text}")
        else:
            st.info("Modelo listo para sugerir cuando inicie la mano.")
    with rec_cols[1]:
        if st.session_state.last_basic:
            st.caption(f"Estrategia básica: {st.session_state.last_basic}")
        else:
            st.caption("Estrategia básica disponible cuando haya cartas.")

    if st.session_state.result_text:
        st.write("")
        st.markdown(f"**Resultado:** {st.session_state.result_text}")
    elif st.session_state.round_over and not st.session_state.player_cards_values:
        st.info("Presiona *Nueva mano* para comenzar una ronda.")

with tab_heatmap:
    st.subheader("Heatmap de acciones recomendadas")
    st.caption("Se consulta al modelo para cada combinacion de total del jugador y carta visible del dealer.")

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
        action_palette = {"hit": "#4CAF50", "stand": "#2196F3"}
        chart_height = max(300, min(len(totals) * 35, 900))
        chart_heatmap = (
            alt.Chart(grid)
            .mark_rect()
            .encode(
                x=alt.X("dealer_visible_card:N", title="Carta visible del dealer", sort=DEALER_RANKS),
                y=alt.Y(
                    "player_total:O",
                    title="Suma de cartas del jugador",
                    sort=totals,
                    axis=alt.Axis(labelAngle=0),
                ),
                color=alt.Color(
                    "pred_action:N",
                    title="Accion recomendada",
                    scale=alt.Scale(domain=list(action_palette.keys()), range=list(action_palette.values())),
                ),
                tooltip=["player_total", "dealer_visible_card", "pred_action"],
            )
            .properties(height=chart_height)
        )
        st.altair_chart(chart_heatmap, use_container_width=True)

        with st.expander("Ver datos del heatmap"):
            st.dataframe(grid, use_container_width=True)

        st.markdown("### Comparacion con estrategia basica (dos cartas iniciales)")
        try:
            two_card_hands = [
                ",".join(cards)
                for cards in itertools.combinations_with_replacement(DEALER_RANKS, 2)
            ]

            rows = []
            for hand_csv in two_card_hands:
                cards = parse_cards(hand_csv)
                for dealer in DEALER_RANKS:
                    rows.append(
                        {
                            "player_cards": hand_csv,
                            "player_cards_step": hand_csv,
                            "dealer_cards": dealer,
                            "dealer_visible_card": dealer,
                            "actions": "",
                            "step": 1,
                            "player_total": hand_value(hand_csv),
                            "cards_remaining": 0,
                            "decks_remaining": 0,
                            "running_count_end": 0,
                            "true_count_end": 0,
                            "true_count_prev_round": 0,
                            "bet_amount": 0,
                            "final_result": "",
                        }
                    )

            df_hands = pd.DataFrame(rows)
            df_hands["pred_action"] = model.predict(df_hands[["player_total", "dealer_visible_card"]])

            def basic_for_row(row: pd.Series) -> str:
                cards = parse_cards(row["player_cards"])
                return basic_strategy(HandState(cards, step=int(row.get("step", 1))), row["dealer_visible_card"])

            df_hands["basic_action"] = df_hands.apply(basic_for_row, axis=1)
            df_hands["basic_action"] = df_hands["basic_action"].replace({"double": "hit"})
            df_hands = df_hands[df_hands["basic_action"].isin(["hit", "stand"])].reset_index(drop=True)
            df_hands["agree"] = df_hands["pred_action"] == df_hands["basic_action"]

            conf = (
                df_hands.groupby(["basic_action", "pred_action"])
                .size()
                .reset_index(name="count")
            )
            chart_confusion = (
                alt.Chart(conf)
                .mark_rect()
                .encode(
                    x=alt.X("pred_action:N", title="Accion del modelo"),
                    y=alt.Y("basic_action:N", title="Accion estrategia basica"),
                    color=alt.Color("count:Q", title="Cantidad"),
                    tooltip=["basic_action", "pred_action", "count"],
                )
                .properties(title="Confusion: Modelo vs Estrategia Basica", width=380, height=320)
            )

            summary = (
                df_hands.groupby("player_total")
                .agg(total=("agree", "size"), agree_sum=("agree", "sum"))
                .reset_index()
            )
            summary["disagree_rate"] = 1 - summary["agree_sum"] / summary["total"]
            top_disagree = summary.sort_values("disagree_rate", ascending=False).head(12)

            chart_disagree = (
                alt.Chart(top_disagree)
                .mark_bar()
                .encode(
                    x=alt.X("disagree_rate:Q", title="Tasa de desacuerdo", axis=alt.Axis(format=".0%")),
                    y=alt.Y(
                        "player_total:O",
                        sort="-x",
                        title="Suma del jugador",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    tooltip=["player_total", alt.Tooltip("disagree_rate:Q", format=".2f"), "total"],
                    color=alt.Color(
                        "disagree_rate:Q",
                        scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                    ),
                )
                .properties(
                    title="Totales iniciales con mayor desacuerdo (modelo vs basica)",
                    width=600,
                    height=450,
                )
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.altair_chart(chart_confusion, use_container_width=True)
            with col_right:
                st.altair_chart(chart_disagree, use_container_width=True)

            with st.expander("Ver datos del analisis (primeras filas)"):
                st.dataframe(df_hands.head(200), use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo generar el analisis de acuerdo: {exc}")
