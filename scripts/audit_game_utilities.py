from pathlib import Path
import hashlib
import json
import argparse

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Audit utility sums at every terminal history.")
parser.add_argument("--input", type=Path, default=REPO / "data/derived/game_sizes.json")
parser.add_argument("--output", type=Path, default=REPO / "outputs/stats/game_utilities.json")
args = parser.parse_args()
import pyspiel
source = args.input
args.output.parent.mkdir(parents=True, exist_ok=True)
def portable_path(path):
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name
configs = json.loads(source.read_text(encoding='utf-8'))
results = []
for config in configs:
    game = pyspiel.load_game_as_turn_based(config['definition'])
    stack = [game.new_initial_state()]
    sums = set()
    examples = {}
    terminals = 0
    while stack:
        state = stack.pop()
        if state.is_terminal():
            terminals += 1
            returns = state.returns()
            total = round(sum(returns), 12)
            sums.add(total)
            examples.setdefault(str(total), returns)
        elif state.is_chance_node():
            stack.extend(state.child(a) for a, p in state.chance_outcomes() if p > 0)
        else:
            stack.extend(state.child(a) for a in state.legal_actions())
    assert terminals == config['terminal_histories']
    result = {
        'game': config['game'],
        'definition': config['definition'],
        'declared_utility_type': str(game.get_type().utility),
        'terminal_count': terminals,
        'terminal_utility_sums': sorted(sums),
        'example_returns_by_sum': examples,
        'zero_sum': sums == {0.0},
        'constant_sum': len(sums) == 1,
        'nonconstant_sum': len(sums) > 1,
    }
    results.append(result)
    print(json.dumps(result), flush=True)
args.output.write_text(json.dumps({
    'source': portable_path(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'pyspiel_sha256': hashlib.sha256(Path(pyspiel.__file__).read_bytes()).hexdigest(),
    'results': results,
}, indent=2) + '\n', encoding='utf-8')
