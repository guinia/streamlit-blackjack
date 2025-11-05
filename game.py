"""
Core game logic for simulating rounds of Blackjack without suits and with
Hi‑Lo counting.

This module defines the classes and functions necessary to model a shoe,
hands, and the flow of a Blackjack round.  Card ranks are tracked without
their suits; each deck contributes four copies of each rank into the shoe.
The probability of drawing each rank therefore diminishes appropriately
when a particular rank is dealt.

The simulation supports basic game actions: hit, stand, double, and split.
It updates the running Hi‑Lo count after each round and provides the true
count used to adjust bets for subsequent rounds.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import random



# ----- Card definitions -----
RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
VALUES = {
    "A": 11,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}


def hi_lo_value(rank: str) -> int:
    """Return the Hi‑Lo count value for a given rank."""
    if rank in ("2", "3", "4", "5", "6"):
        return 1
    if rank in ("7", "8", "9"):
        return 0
    return -1  # 10, J, Q, K, A


class Shoe:
    """
    A shoe containing multiple decks of 52 cards, but only storing card
    ranks (suits are omitted).  Each rank appears four times per deck.

    The shoe reshuffles automatically when 75%% of its cards have been
    dealt (i.e. when only 25%% remain), resetting the running Hi‑Lo count.
    """

    def __init__(self, num_decks: int = 6, shuffle_seed: Optional[int] = None) -> None:
        self.num_decks = num_decks
        self._rng = random.Random(shuffle_seed)
        self._cards_total = 0
        self._shoe: List[str] = []
        self.reshuffle()

    def reshuffle(self) -> None:
        """Reshuffle the shoe by loading all decks and randomising the order."""
        self._shoe = []
        for _ in range(self.num_decks):
            for rank in RANKS:
                # 4 copies per rank per deck
                self._shoe.extend([rank] * 4)
        self._rng.shuffle(self._shoe)
        self._cards_total = len(self._shoe)

    def cards_remaining(self) -> int:
        return len(self._shoe)

    def decks_remaining(self) -> float:
        return max(0.0001, len(self._shoe) / 52.0)

    def pop(self) -> str:
        """
        Pop one card rank from the shoe.  If fewer than 25%% of cards remain,
        the shoe is automatically reshuffled.
        
        if len(self._shoe) <= 0.25 * self._cards_total:
            self.reshuffle()
        """
        return self._shoe.pop()


@dataclass
class HandState:
    """Represents a player's or dealer's hand and the associated bet."""

    cards: List[str]
    bet: float
    actions: List[str]
    doubled: bool = False

    @property
    def value(self) -> int:
        """
        Compute the Blackjack value of the hand, counting Aces as 1 or 11
        to avoid busting where possible.
        """
        total = 0
        aces = 0
        for r in self.cards:
            if r == "A":
                total += 11
                aces += 1
            else:
                total += VALUES[r]
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21

    def can_split(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def can_double(self) -> bool:
        return len(self.cards) == 2


StrategyFn = Callable[[HandState, str], str]


@dataclass
class RoundResult:
    """
    Represents the outcome of a single hand (possibly one of several from
    splits) within a round of Blackjack.
    """

    round_id: int
    hand_number: int
    player_cards: List[str]
    dealer_cards: List[str]
    actions: List[str]
    bet_amount: float
    final_result: str  # "win" | "lose" | "push"
    blackjack: bool
    busted: bool
    strategy_used: str
    bet_mode: str
    true_count_prev_round: float
    running_count_end: float
    true_count_end: float
    cards_remaining: int
    decks_remaining: float


def _dealer_play(dealer: HandState, shoe: Shoe) -> None:
    """Dealer hits until reaching 17 or higher."""
    while dealer.value < 17:
        dealer.cards.append(shoe.pop())
        dealer.actions.append("hit")
    dealer.actions.append("stand")


def _settle(player_value: int, dealer_value: int) -> int:
    """
    Compare player and dealer totals returning:
        +1 for a win, 0 for a push, and −1 for a loss.
    """
    if player_value > 21:
        return -1
    if dealer_value > 21:
        return 1
    if player_value > dealer_value:
        return 1
    if player_value < dealer_value:
        return -1
    return 0


def _bet_from_true_count(tc: float, base: float = 10.0, max_mult: int = 5) -> float:
    """
    Compute the bet multiplier from the true count using a simple ramp:
      tc ≤ 0 → base bet
      tc = 1 → base×2, tc = 2 → base×3, tc = 3 → base×4, tc ≥ 4 → base×5
    """
    if tc <= 0:
        return float(base)
    mult = 1 + int(tc)
    if mult > max_mult:
        mult = max_mult
    return float(base * mult)

def legal_actions(hand: HandState) -> List[str]:
    acts = ["stand", "hit"]
    if hand.can_double():
        acts.append("double")
    # Si querés incluir split en esta v1, descomentá:
    # if hand.can_split():
    #     acts.append("split")
    return acts

def clone_shoe(src: Shoe) -> Shoe:
    """Clona el shoe actual (mismo contenido de cartas)."""
    dst = Shoe(num_decks=src.num_decks)     # crea base
    dst._shoe = list(src._shoe)             # copia estado
    dst._cards_total = src._cards_total
    # RNG distinto por rama = orden distinto de draws (bien para Monte Carlo)
    dst._rng = random.Random()              # si querés reproducibilidad, seteá una seed externa
    return dst

def apply_action_once(hand: HandState, action: str, shoe: Shoe) -> Tuple[HandState, bool]:
    """
    Aplica UNA acción del jugador sobre una copia de la mano y devuelve:
    - (nueva_mano, terminal_del_turno_del_jugador)
    """
    h = HandState(cards=list(hand.cards), bet=hand.bet, actions=list(hand.actions), doubled=hand.doubled)

    if action == "hit":
        h.cards.append(shoe.pop())
        h.actions.append("hit")
        # si llega a 21 o bust, el turno del jugador termina
        if h.value >= 21:
            if h.value == 21:
                h.actions.append("stand")
            return h, True
        return h, False

    if action == "double" and h.can_double():
        h.bet *= 2.0
        h.doubled = True
        h.actions.append("double")
        h.cards.append(shoe.pop())
        h.actions.append("stand")
        return h, True  # double cierra el turno

    # "stand" o acción ilegal => stand
    h.actions.append("stand")
    return h, True

def dfs_play_all_paths(
    init_player: List[str],
    init_dealer: List[str],
    shoe_for_node: Shoe,
    base_bet: float,
    round_id: int,
    true_count_prev_round: float,
    bet_mode: str,
    results_acc: List[RoundResult],
):
    """
    Explora en profundidad TODAS las ramas desde la mano inicial.
    En cada nodo: toma acciones LEGALES, baraja su orden al azar y recurre.
    Cuando el jugador termina (stand/double/bust), juega dealer y registra 1 fila.
    """

    # Estado raíz
    root_player = HandState(cards=list(init_player), bet=base_bet, actions=[])
    root_dealer = HandState(cards=list(init_dealer), bet=0.0, actions=[])

    def _dealer_phase_and_emit(player: HandState, dealer: HandState, shoe: Shoe):
        # Dealer juega
        _dealer_play(dealer, shoe)
        # Resultado
        outcome = _settle(player.value, dealer.value)
        final_result = "push" if outcome == 0 else ("win" if outcome > 0 else "lose")
        busted = player.value > 21
        blackjack = player.is_blackjack()

        # Conteo Hi-Lo local de esta rama (cartas vistas)
        seen = player.cards + dealer.cards
        local_rc = sum(hi_lo_value(r) for r in seen)
        local_tc = local_rc / max(0.0001, 52.0 * shoe.num_decks)  # aprox. TC local por rama

        results_acc.append(
            RoundResult(
                round_id=round_id,
                hand_number=1,
                player_cards=list(player.cards),
                dealer_cards=list(dealer.cards),
                actions=list(player.actions),
                bet_amount=player.bet,
                final_result=final_result,
                blackjack=blackjack,
                busted=busted,
                strategy_used="random-dfs",
                bet_mode=bet_mode,
                true_count_prev_round=round(true_count_prev_round, 3),
                running_count_end=round(local_rc, 3),
                true_count_end=round(local_tc, 3),
                cards_remaining=shoe_for_node.cards_remaining(),   # informativo
                decks_remaining=round(shoe_for_node.decks_remaining(), 3),
            )
        )

    def _dfs(player: HandState, dealer: HandState, shoe: Shoe):
        # Si ya bust, termina rama y emite
        if player.value > 21:
            _dealer_phase_and_emit(player, dealer, shoe)
            return

        # Si es blackjack natural, cerrar inmediatamente el turno
        if player.is_blackjack():
            if not player.actions or player.actions[-1] != "stand":
                player.actions.append("stand")
            _dealer_phase_and_emit(player, dealer, shoe)
            return

        # Acciones legales del nodo actual (orden aleatorio)
        acts = legal_actions(player)
        random.shuffle(acts)

        for act in acts:
            # Clonar estado para la rama
            p = HandState(cards=list(player.cards), bet=player.bet, actions=list(player.actions), doubled=player.doubled)
            d = HandState(cards=list(dealer.cards), bet=dealer.bet, actions=list(dealer.actions), doubled=dealer.doubled if hasattr(dealer,'doubled') else False)
            sh = clone_shoe(shoe)

            # Aplicar acción una vez
            p, player_turn_done = apply_action_once(p, act, sh)

            if player_turn_done:
                # emite fila terminal (dealer juega adentro)
                _dealer_phase_and_emit(p, d, sh)
            else:
                # seguir profundizando (más hits posibles)
                _dfs(p, d, sh)

    # arrancar DFS desde la raíz
    _dfs(root_player, root_dealer, clone_shoe(shoe_for_node))

def simulate_rounds(
    rounds: int,
    num_decks: int,
    base_bet: float,
    strategy_name: str,      # se conserva por compatibilidad (no se usa)
    strategy_fn: StrategyFn, # se conserva por compatibilidad (no se usa)
    seed: Optional[int] = None,
    bet_mode: str = "fixed", # "fixed" | "hi-lo"
) -> List[RoundResult]:

    rng = random.Random(seed)
    # Shoe solo para REPARTIR la mano inicial de cada round
    deal_shoe = Shoe(num_decks=num_decks, shuffle_seed=seed)
    cards_total = deal_shoe.cards_remaining()
    results: List[RoundResult] = []
    running_count_prev = 0.0
    true_count_prev = 0.0

    for round_id in range(1, rounds + 1):
        # reshuffle al iniciar round si quedó <25%
        if deal_shoe.cards_remaining() <= 0.25 * cards_total:
            deal_shoe.reshuffle()
            running_count_prev = 0.0
            cards_total = deal_shoe.cards_remaining()
            true_count_prev = 0.0

        # apuesta base según bet_mode
        if bet_mode == "hi-lo":
            bet_base = _bet_from_true_count(true_count_prev, base=base_bet)
        else:
            bet_base = float(base_bet)

        # repartir mano inicial (jugador/dealer)
        init_player = [deal_shoe.pop(), deal_shoe.pop()]
        init_dealer = [deal_shoe.pop(), deal_shoe.pop()]

        # DFS: explora TODAS las ramas, eligiendo orden aleatorio en cada nodo
        dfs_play_all_paths(
            init_player=init_player,
            init_dealer=init_dealer,
            shoe_for_node=deal_shoe,                 # sólo para referenciar decks_remaining/cards_remaining
            base_bet=bet_base,
            round_id=round_id,
            true_count_prev_round=true_count_prev,
            bet_mode=bet_mode,
            results_acc=results,
        )

        # avanzar el conteo “global” SOLO con el reparto inicial (coherente con tu idea de TC por ronda)
        running_count_prev += sum(hi_lo_value(r) for r in (init_player + init_dealer))
        true_count_prev = running_count_prev / deal_shoe.decks_remaining()

    return results