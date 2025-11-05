"""
Modelo simplificado para decidir entre 'hit' y 'stand' en Blackjack.

Entrenamiento basado exclusivamente en:
    - Suma total de cartas del jugador antes de cada accion.
    - Primera carta visible del dealer.

Las partidas ganadas y perdidas se incluyen por igual. Se reemplaza 'double'
por 'hit', se descartan 'split' y se etiqueta cada estado con la accion que
históricamente tuvo mejor tasa de victorias (usando `final_result` como
indicador de buen/mal resultado). El modelo final únicamente aprende a predecir
entre 'hit' y 'stand'. Finalmente, se exporta el pipeline entrenado a
`models/blackjack_action_model.joblib` para su uso en la app Streamlit.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_components import expand_dataset, hand_value

# --- Rutas de entrada/salida -------------------------------------------------
DATA_PATH = Path("Simulacion_BJ.csv")
MODEL_PATH = Path("models/blackjack_action_model.joblib")

# --- Helpers -----------------------------------------------------------------
FACE_AS_TEN = {"J": "10", "Q": "10", "K": "10"}


# --- Rutas de entrada/salida -------------------------------------------------
def load_and_expand(path: Path) -> pd.DataFrame:
    """Carga el CSV original y expande cada accion en una fila."""
    df_raw = pd.read_csv(path)
    expanded = expand_dataset(df_raw)
    return expanded


def prepare_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset de estados para entrenamiento:
      * Asegura acciones en minuscula.
      * Reemplaza 'double' por 'hit' y elimina 'split'.
      * Calcula total de cartas del jugador y bandera de victoria.
      * Normaliza la carta visible del dealer mapeando J/Q/K a '10'.
      * Conserva solo columnas relevantes.
    """
    states = df.copy()
    states["action"] = states["action"].fillna("").str.strip().str.lower()

    # Sustituir 'double' por 'hit' y descartar 'split'
    states.loc[states["action"] == "double", "action"] = "hit"
    states = states[states["action"].isin(["hit", "stand"])]

    # Calculo de total del jugador (usamos player_cards_step porque refleja el
    # estado exacto antes de la accion).
    states["player_cards_step"] = states["player_cards_step"].fillna(states["player_cards"])
    states["player_total"] = states["player_cards_step"].apply(hand_value)

    # Primera carta visible del dealer, normalizando figuras como 10.
    states["dealer_visible_card"] = (
        states["dealer_visible_card"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(FACE_AS_TEN)
    )
    states = states[states["dealer_visible_card"] != ""]

    # Indicador de partida ganada/perdida
    states["win_flag"] = states["final_result"].fillna("").str.lower().eq("win").astype(int)

    return states[
        [
            "game_id",
            "player_total",
            "dealer_visible_card",
            "action",
            "win_flag",
        ]
    ].copy()


def build_best_action_labels(states: pd.DataFrame) -> pd.DataFrame:
    """
    Determina la mejor accion ('hit' o 'stand') por combinacion
    (player_total, dealer_visible_card) en funcion de la tasa de victorias.
    """
    grouped = (
        states.groupby(["player_total", "dealer_visible_card", "action"], as_index=False)
        .agg(win_rate=("win_flag", "mean"), samples=("win_flag", "size"))
    )

    if grouped.empty:
        raise ValueError("No hay acciones disponibles tras el filtrado.")

    # Ordenar para quedarnos con la mejor accion: mayor win_rate, luego mas muestras.
    grouped = grouped.sort_values(
        by=["player_total", "dealer_visible_card", "win_rate", "samples"],
        ascending=[True, True, False, False],
    )

    best = (
        grouped.groupby(["player_total", "dealer_visible_card"], as_index=False)
        .first()
        .rename(columns={"action": "best_action"})
    )

    return best[["player_total", "dealer_visible_card", "best_action"]]


def make_training_frame(states: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Une los estados con la mejor accion identificada.
    Cada estado hereda la accion recomendada para su combinacion total/dealer.
    """
    merged = states.merge(
        labels,
        on=["player_total", "dealer_visible_card"],
        how="inner",
    )
    return merged


def build_pipeline() -> Pipeline:
    """Pipeline: one-hot para carta del dealer + LogisticRegression."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["player_total"]),
            ("dealer", OneHotEncoder(handle_unknown="ignore"), ["dealer_visible_card"]),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    # 1) Cargar y expandir dataset original.
    df_expanded = load_and_expand(DATA_PATH)
    print(f"Dataset expandido: {df_expanded.shape[0]} filas")

    # 2) Construir estados filtrados.
    states = prepare_states(df_expanded)
    print(f"Estados tras limpieza: {states.shape[0]} filas")

    # 3) Determinar la mejor accion por combinacion de total/carta del dealer.
    best_action_map = build_best_action_labels(states)
    print(f"Combinaciones distintas evaluadas: {best_action_map.shape[0]}")

    # 4) Dataset final para entrenamiento (solo stand/hit).
    training_df = make_training_frame(states, best_action_map)
    X = training_df[["player_total", "dealer_visible_card"]].copy()
    y = training_df["best_action"].copy()

    # 5) Split train/test (sin final_result en features).
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]} muestras | Test size: {X_test.shape[0]} muestras")

    # 6) Entrenar pipeline.
    model = build_pipeline()
    model.fit(X_train, y_train)

    # 7) Evaluar en test.
    y_pred = model.predict(X_test)
    print("\nEvaluacion en test (solo stand/hit):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))

    # 8) Guardar modelo entrenado.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
