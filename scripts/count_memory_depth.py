from pathlib import Path
import hashlib
import json
import argparse

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Count own predecessor decisions at information sets.")
parser.add_argument("--input", type=Path, default=REPO / "data/derived/game_sizes.json")
parser.add_argument("--output", type=Path, default=REPO / "outputs/stats/memory_depths.json")
args = parser.parse_args()
import pyspiel
source = args.input
args.output.parent.mkdir(parents=True, exist_ok=True)
def portable_path(path):
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name
baseline = json.loads(source.read_text(encoding='utf-8'))
results = []
for expected in baseline:
    game = pyspiel.load_game_as_turn_based(expected['definition'])
    per_player = [dict() for _ in range(game.num_players())]
    stack = [(game.new_initial_state(), (0,) * game.num_players())]
    histories = terminals = 0
    while stack:
        state, counts = stack.pop()
        histories += 1
        if state.is_terminal():
            terminals += 1
            continue
        if state.is_chance_node():
            for action, probability in state.chance_outcomes():
                assert probability > 0
                stack.append((state.child(action), counts))
            continue
        player = state.current_player()
        assert 0 <= player < game.num_players()
        info = state.information_state_string(player)
        depth = counts[player]
        if info in per_player[player]:
            assert per_player[player][info] == depth
        else:
            per_player[player][info] = depth
        next_counts = list(counts)
        next_counts[player] += 1
        next_counts = tuple(next_counts)
        for action in state.legal_actions():
            stack.append((state.child(action), next_counts))
    assert histories == expected['histories']
    assert terminals == expected['terminal_histories']
    assert [len(x) for x in per_player] == expected['information_sets_per_player']
    maxima = [max(x.values(), default=0) for x in per_player]
    result = {
        'game': expected['game'],
        'definition': expected['definition'],
        'players': game.num_players(),
        'information_sets': sum(map(len, per_player)),
        'states': histories,
        'terminals': terminals,
        'memory_depth_per_player': maxima,
        'memory_depth': max(maxima),
        'information_set_depths_consistent': True,
    }
    results.append(result)
    print(json.dumps(result), flush=True)

output = {
    'source': portable_path(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'depth_definition': 'max d_I: own predecessor decisions before the current information set; excludes current decision and chance transitions',
    'pyspiel_sha256': hashlib.sha256(Path(pyspiel.__file__).read_bytes()).hexdigest(),
    'results': results,
}
args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')

decision_output = args.output.with_name("decision_counts.json")
decision_output.write_text(json.dumps({
    "source": portable_path(args.output),
    "source_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    "definition": "1 + max d_I; maximum decisions made by a single player along a play, including the current decision and excluding chance transitions",
    "recommendation_limit": "At most one observed recommendation per own decision",
    "results": [{"game": item["game"], "max_player_decisions": item["memory_depth"] + 1} for item in results],
}, indent=2) + "\n", encoding="utf-8")
