"""Game titles used by the paper figures."""
from hr_edl_data import experiment_parameters as xp
_game_label_map = {"sheriff": "Sheriff", "kuhn_3p": "Kuhn poker", "leduc": "Leduc hold'em", "random_goofspiel": "goofspiel", "battleship": "Battleship"}

def game_label(tag, t=None):
  if tag in _game_label_map:
    name = _game_label_map[tag]
  elif tag[:len("goofspiel")] == "goofspiel":
    name = "goofspiel"
  else:
    name = tag

  params_string = ""
  if tag in xp.NUM_PLAYERS_MAP:
    params_string = r"N=\num{{{}}}".format(xp.NUM_PLAYERS_MAP[tag])
  if t is not None:
    params_string = ",".join(filter(None, [params_string, r"T=\num{{{}}}".format(t)]))
  if tag in xp.EXTRA_GAME_PARAMS_MAP:
    params = tuple(p for p in xp.EXTRA_GAME_PARAMS_MAP[tag] + (params_string,) if p)
    return "{}$({})$".format(name, ",".join(params))
  if params_string != "":
    return "{}$({})$".format(name, params_string)
  return name
