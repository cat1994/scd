# Release validation

The extraction preserves the original numerical records and separates them
from freshly generated validation output. The full 1000-round tournaments were
not rerun during packaging. The subsequent SCD naming migration changes labels,
file names and source identifiers; all numerical tuple text is preserved.

## Original packaging checks

The checks below describe the original packaging validation, before the naming
migration. The corresponding C++ and Python evidence retains the original
engine fingerprints and environment details.

- All 28 raw SSV files match their archived SHA-256 hashes exactly.
- All 518,000 parsed records are retained, including auxiliary BR rows.
- All 28 first-party C++ source files match their archived source fingerprints.
- A fresh standalone Release build succeeded with GNU C++ 11.4.0 and GNU C++17.
  All seven benchmark definitions loaded, and the two auxiliary C++ tools ran.
- Three retained C++ scientific checks passed: local coverage, tabular learner,
  and action transformations. Five Python checks passed, covering complete
  matrices, missing records, configuration compatibility and protected output.
- Kuhn and Leduc, main and all three ablations, were rerun for five rounds:
  740 tuples had identical payoff and both regret fields compared with the
  archived data. Runtime and diagnostic-time fields are intentionally excluded
  from that equality comparison.
- The main and ablation data-storage commands saved 392,000 and 126,000 records.
  All three seven-game figures regenerated successfully and their PNG renders
  were visually reviewed for labels, cropping and layout.
- All 91 endpoints were recomputed from raw files. All 273 formatted table
  cells and both complete LaTeX fragments match the manuscript. Numerical
  differences before formatting are at floating-point summation precision
  (maximum displayed-payoff difference below 3e-12).
- The seven-game size, depth, decision-count and terminal-utility statistics
  were rerun using the existing Python OpenSpiel 1.4 environment and matched
  the archived scientific fields.
- Short launcher runs, reuse with `--skip-existing`, and refusal to overwrite
  results without that flag were exercised. New metadata contains relative
  paths and content fingerprints, without Git HEAD or author identity.
- Machine paths and private host markers were scanned in the distributable
  files. Reference figure PDFs contain plotting-software and creation-time
  metadata, with no author field. Public dependency copyright names are kept.

Detailed evidence is in `data/derived/cpp_validation.json`,
`python_validation.json`, `table_validation.json` and
`game_statistics_validation.json`.

## SCD naming migration

The compressed and uncompressed methods now use `k-SCD` and
`k-SCD_UNCOMPRESSED` throughout source code, raw records, sidecars and derived
tables. Experiment prefixes are `scd_main` and `scd_ablation`; the k parameter
is `scd_budget`. No update rule or numeric experiment setting is changed.

All 28 migrated SSV files were checked against the expected bytes obtained by
replacing algorithm labels in the original files only. The original hashes
are retained as `original_sha256` in `data/provenance.json` and
`original_output_sha256` in the sidecars. Current raw hashes, byte counts and
derived-data references are updated together. Historical source and executable
fingerprints still identify the archived runs; the current release manifest
identifies the renamed source files.

Current migration checks are recorded in `data/derived/scd_naming_validation.json`.

## Reproduction boundaries

The default build preserves `-O3 -march=native -funroll-loops -DNDEBUG`, which
was necessary for the checked Leduc trajectory. An initial generic build
without native optimization differed from the archive in 12 of 280 Leduc
main tuples; restoring the original flags eliminated those differences.
Other compilers or CPUs can still affect decisions near numerical regret ties.
The archive provides the original full-run data independently of these effects.

One old generic `policy_evaluation_test` is outside the extracted suite: it
failed on the original preexisting binary and on the extracted build with the
same assertion (`450 != 3780`). No scientific implementation was changed to
bypass it. The failure and retained validation scope are recorded in the C++
validation evidence.

The OpenSpiel source revision is identified by distributed file hashes. Its
original recipe's commit pin is recorded but cannot independently establish the
version of a checkout without its own Git metadata. Dependency notices describe
this distinction. Portable MathText rendering may differ typographically from
the original TeX-rendered reference PDFs; it uses the same numerical curves.
