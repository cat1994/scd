"""The seven paper games and two exact tournament pools.

SCD identifiers distinguish compressed and uncompressed implementations.
"""
from pathlib import Path

GAME_MAP = {'kuhn_3p': 'kuhn_poker(players=3)', 'leduc': 'leduc_poker', 'random_goofspiel': 'goofspiel(imp_info=True,num_cards=4,points_order=random)', 'goofspiel_3p': 'goofspiel(imp_info=True,num_cards=4,points_order=descending,players=3)', 'goofspiel_ascending_3p': 'goofspiel(imp_info=True,num_cards=4,points_order=ascending,players=3)', 'sheriff': 'sheriff', 'battleship': 'battleship(board_width=3,board_height=2,ship_sizes=[1],ship_values=[1.0],num_shots=2,allow_repeated_shots=False,loss_multiplier=1.0)'}

NUM_PLAYERS_MAP = {'kuhn_3p': 3, 'leduc': 2, 'random_goofspiel': 2, 'goofspiel_3p': 3, 'goofspiel_ascending_3p': 3, 'sheriff': 2, 'battleship': 2}

EXTRA_GAME_PARAMS_MAP = {'random_goofspiel': ('4', 'R'), 'goofspiel_3p': ('4', '\\downarrow'), 'goofspiel_ascending_3p': ('4', '\\uparrow')}

SCD_MAIN_ALGORITHMS = ('CFR', '1-SCD', '2-SCD', '4-SCD', 'A-EFR_IN', 'CSPS-EFR', 'BEHAV-EFR')

ALG_GROUP_MAP = {'scd_main': 11, 'scd_ablation': 99}

class ExperimentParameters:
    """Parse the published SCD SSV filename convention."""
    def __init__(self, filename):
        parts = Path(filename).name.split(".")
        if len(parts) != 7 or parts[0] not in ALG_GROUP_MAP or parts[1] not in GAME_MAP or parts[2:5] != ["null", "fixed", "gen"] or parts[6] != "ssv":
            raise ValueError(f"Unsupported SCD result filename: {Path(filename).name}")
        self.alg_group_str, self.game_tag, self.mode = parts[0], parts[1], parts[3]
        self.scd_budget = int(parts[5])
        if self.scd_budget not in (1, 2, 4) or (self.alg_group_str == "scd_main" and self.scd_budget != 4):
            raise ValueError(f"Unsupported SCD budget: {self.scd_budget}")
