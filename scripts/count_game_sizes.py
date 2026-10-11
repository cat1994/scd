#!/usr/bin/env python3
"""Enumerate the seven benchmark instances used in the paper."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
CONFIGS = (
    ("kuhn_3p", "kuhn_poker(players=3)"),
    ("leduc", "leduc_poker"),
    ("random_goofspiel", "goofspiel(imp_info=True,num_cards=4,points_order=random)"),
    ("goofspiel_3p", "goofspiel(imp_info=True,num_cards=4,points_order=descending,players=3)"),
    ("goofspiel_ascending_3p", "goofspiel(imp_info=True,num_cards=4,points_order=ascending,players=3)"),
    ("sheriff", "sheriff"),
    ("battleship", "battleship(board_width=3,board_height=2,ship_sizes=[1],ship_values=[1.0],num_shots=2,allow_repeated_shots=False,loss_multiplier=1.0)"),
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/stats/game_sizes.json")
    args = parser.parse_args()
    import pyspiel
    results = []
    for name, definition in CONFIGS:
        game = pyspiel.load_game_as_turn_based(definition)
        seen = [set() for _ in range(game.num_players())]
        nontrivial = [set() for _ in range(game.num_players())]
        hist = term = chance = max_depth = 0
        stack = [(game.new_initial_state(), 0)]
        started = time.monotonic()
        while stack:
            state, depth = stack.pop()
            hist += 1
            max_depth = max(max_depth, depth)
            if state.is_terminal():
                term += 1
                continue
            if state.is_chance_node():
                chance += 1
                for action, prob in state.chance_outcomes():
                    assert prob > 0
                    stack.append((state.child(action), depth + 1))
                continue
            player = state.current_player()
            assert 0 <= player < game.num_players()
            key = state.information_state_string(player)
            actions = state.legal_actions()
            seen[player].add(key)
            if len(actions) > 1:
                nontrivial[player].add(key)
            stack.extend((state.child(action), depth + 1) for action in actions)
        result = {
            "game": name, "definition": definition,
            "parameters": {k: str(v) for k, v in game.get_parameters().items()},
            "players": game.num_players(),
            "information_sets_per_player": [len(x) for x in seen],
            "information_sets": sum(map(len, seen)),
            "nontrivial_information_sets_per_player": [len(x) for x in nontrivial],
            "terminal_histories": term, "histories": hist, "chance_histories": chance,
            "max_depth": max_depth, "enumeration_seconds": time.monotonic() - started,
        }
        results.append(result)
        print(f"{name}: {result['information_sets']} information sets, {hist} states, {term} terminals")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    try:
        version = importlib.metadata.version("open-spiel")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    args.output.with_name("game_sizes_environment.json").write_text(
        json.dumps({"packages": {"open-spiel": version}}, indent=2) + "\n",
        encoding="utf-8")

if __name__ == "__main__":
    main()
