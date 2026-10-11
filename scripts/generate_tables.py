"""Rebuild the paper's endpoint tables directly from archived raw SSV runs.

OSR is already divided by the number of updates by the raw-data reader.
Only its displayed table cells are multiplied by 1000 (10^-3 units).
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from hr_edl_data.scd_analysis import (
    GAMES, MAIN_POOL, ablation_pool, curve_data, load_raw_data,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--main-data", type=Path, default=REPO / "data/raw/main")
parser.add_argument("--ablation-data", type=Path, default=REPO / "data/raw/ablation")
parser.add_argument("--output", type=Path, default=REPO / "outputs/tables")
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
audit = {"runs": []}
source_records = []

def portable_path(path):
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name

for family, directory in (("main", args.main_data), ("ablation", args.ablation_data)):
    frame = load_raw_data(directory, require_metadata=True)
    for game in GAMES:
        for k in ((4,) if family == "main" else (1, 2, 4)):
            pool = tuple(MAIN_POOL) if family == "main" else ablation_pool(k)
            curves = curve_data(frame, game, pool, pool,
                                budget=None if family == "main" else k,
                                metrics=("osr",))
            raw_paths = sorted(directory.rglob(f"*.{game}.null.fixed.gen.{k}.ssv"))
            if len(raw_paths) != 1:
                raise ValueError(f"Expected one archived raw file for {family}/{game}/k={k}, got {raw_paths}")
            raw_path = raw_paths[0]
            raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            selected = frame[(frame.game_tag == game) & frame.row_alg.isin(pool)]
            if family == "ablation":
                selected = selected[selected.scd_budget == k]
            run_ids = selected.run_id.unique().tolist()
            if len(run_ids) != 1:
                raise ValueError(f"Expected one deterministic archived run, got {run_ids}")
            stats = {}
            for method, curve in curves.items():
                if len(curve) != 1000:
                    raise ValueError(f"{family}/{game}/{method}: expected 1000 rounds, got {len(curve)}")
                stats[method] = {
                    "n": len(curve),
                    "osr_at": {"1000": float(curve.osr.iloc[-1])},
                    "payoff": float(curve.value.iloc[-1]),
                    "seconds": float(curve.runtime_seconds.iloc[-1]),
                }
            audit["runs"].append({
                "family": family, "game": game, "k": k,
                "run_id": run_ids[0], "status": "complete", "return_code": 0,
                "config": {"iterations": 1000},
                "data_sha256": raw_hash, "stats": stats,
            })
            source_records.append({"path": portable_path(raw_path), "sha256": raw_hash})
order=['kuhn_3p','leduc','random_goofspiel','goofspiel_3p','goofspiel_ascending_3p','sheriff','battleship']
names={'kuhn_3p':'Kuhn Poker','leduc':'Leduc Poker','random_goofspiel':'Goofspiel (4, R)','goofspiel_3p':r'Goofspiel (4, $\downarrow$)','goofspiel_ascending_3p':r'Goofspiel (4, $\uparrow$)','sheriff':'Sheriff','battleship':'Battleship'}
methods=['CFR','1-SCD','2-SCD','4-SCD','A-EFR_IN','CSPS-EFR','BEHAV-EFR']
labels=[r'\(\mathrm{SCD}_0\)',r'\(\mathrm{SCD}_1\)',r'\(\mathrm{SCD}_2\)',r'\(\mathrm{SCD}_4\)',r'\(\mathrm{ACT}_{\rm IN}\)','CSPS','BHV']
runs={}
for r in audit['runs']:
    if r['game'] not in order:continue
    assert r['status']=='complete' and r['return_code']==0 and r['config']['iterations']==1000
    runs[(r['family'],r['game'],r['k'])]=r

def transform_value(name,value):
    if name in ('kuhn_3p','leduc'):return 1000*value
    if 'goofspiel' in name:return 50*(value+1)
    return value

def fmt_value(name,value):
    return f'{transform_value(name,value):.2f}' if name in ('kuhn_3p','leduc') or 'goofspiel' in name else f'{value:.5f}'

data=[]
def metric_cells(run,method):
    stat=run['stats'][method]
    assert stat['n']==1000 and '1000' in stat['osr_at']
    osr=stat['osr_at']['1000'];value=stat['payoff'];time=stat['seconds']
    record={'family':run['family'],'game':run['game'],'k':run['k'],'implementation':method,'osr1000':osr,'payoff_native':value,'value_display_scale':transform_value(run['game'],value),'runtime_seconds':time,'osr_cell':f'{1000*osr:.3f}','value_cell':fmt_value(run['game'],value),'runtime_cell':f'{time:.2f}','run_id':run['run_id'],'data_sha256':run['data_sha256']}
    data.append(record)
    return [record['osr_cell'],record['value_cell'],record['runtime_cell']]

main=[r'\begin{table*}[t]',r'\centering',r'\caption{Performance after 1000 rounds in the main comparison, averaged over all seven independent self-play environments. OSR gap is the local convergence statistic in Eq.~\eqref{eq:experimental-local-osr}, with cumulative regret divided by \(1000\) and displayed in units of \(10^{-3}\). Value is cumulative average payoff: mbb for poker, rescaled utility score (\%) for Goofspiel, and native utility for Sheriff and Battleship. Runtime is cumulative learner runtime in seconds.}',r'\label{tab:main-endpoints}',r'\fontsize{8.5}{11}\selectfont',r'\renewcommand{\arraystretch}{1.08}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}',r'\hline','Game & Metric & '+' & '.join(labels)+r'\\',r'\hline']
for game in order:
    run=next(r for r in runs.values() if r['family']=='main' and r['game']==game)
    cells=[metric_cells(run,m) for m in methods]
    for index,metric in enumerate([r'OSR gap \((\cdot10^{-3})\)','Value','Runtime (s)']):
        main.append((names[game] if index==0 else '')+' & '+metric+' & '+' & '.join(x[index] for x in cells)+r'\\')
    main.append(r'\hline')
main.extend([r'\end{tabular*}',r'\end{table*}'])

ablation=[r'\begin{table*}[t]',r'\centering',r'\caption{Performance after 1000 rounds in the compression ablation. Each budget uses its own two-method opponent pool, including the independent self-play environments of both methods. OSR gap is the maximum positive cumulative local regret divided by \(1000\) and averaged over the two environments, displayed in units of \(10^{-3}\); Value uses the same game-specific cumulative-payoff scales as Table~\ref{tab:main-endpoints}; Runtime is cumulative learner time in seconds. The main and ablation tables use different opponent pools.}',r'\label{tab:ablation-endpoints}',r'\fontsize{8.5}{11}\selectfont',r'\renewcommand{\arraystretch}{1.08}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrrr@{}}',r'\hline',r'Game & \(k\) & \multicolumn{3}{c}{Compressed \(\mathrm{SCD}_k\)} & \multicolumn{3}{c}{Uncompressed \(\mathrm{SCD}_{\le k}\)}\\',r'\cline{3-5}\cline{6-8}',r' & & OSR gap \((\cdot10^{-3})\) & Value & Runtime (s) & OSR gap \((\cdot10^{-3})\) & Value & Runtime (s)\\',r'\hline']
for game in order:
    for k in [1,2,4]:
        run=runs[('ablation',game,k)]
        cells=metric_cells(run,f'{k}-SCD')+metric_cells(run,f'{k}-SCD_UNCOMPRESSED')
        ablation.append((names[game] if k==1 else '')+' & '+str(k)+' & '+' & '.join(cells)+r'\\')
    ablation.append(r'\hline')
ablation.extend([r'\end{tabular*}',r'\end{table*}'])
(root/'main_table.tex').write_text('\n'.join(main)+'\n',encoding='utf-8')
(root/'ablation_table.tex').write_text('\n'.join(ablation)+'\n',encoding='utf-8')
assert len(data)==91
(root/'endpoint_data.json').write_text(json.dumps({'sources':source_records,'normalization':'cumulative regret / updates; OSR table cells use 10^-3 units','results':data},indent=2),encoding='utf-8')
with (root/'endpoint_data.csv').open('w',encoding='utf-8',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(data[0]),lineterminator="\n");writer.writeheader();writer.writerows(data)
print('Created 49 main and 42 ablation endpoint entries and two complete result tables.')
for game in order:
    print(game)
    for record in data:
        if record['family']=='ablation' and record['game']==game:print(record['k'],record['implementation'],record['osr_cell'],record['value_cell'],record['runtime_cell'])
