# Third-party notices

The inherited EFR / hindsight-rationality implementation is distributed under
the MIT license in `LICENSE.txt`. Its original copyright notice is preserved.
This is attribution to upstream code, not a statement about manuscript authors.

The release vendors the following dependencies without their repository
histories. License files and source copyright notices remain included.

| Component | Snapshot information | License location |
| --- | --- | --- |
| OpenSpiel | Source subset extracted from the experiment tree; the original build recipe specifies revision `53eeb6127578f91e88a6c0451983d10e3509446d`, but this checkout has no independent Git metadata to prove that revision. File checksums in `SHA256SUMS` identify the actual distributed snapshot. | `hr_edl/src/open_spiel/LICENSE` (Apache 2.0) and per-file notices |
| Abseil | `20200923.3`, revision `6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c` | `hr_edl/src/open_spiel/abseil-cpp/LICENSE` (Apache 2.0) |
| Eigen | Revision `39ec31c0adbdde6b8cda36b3415e9cc2af20dab6`, source version macros 3.3.90 | `hr_edl/src/eigen/COPYING.README`, `COPYING.MPL2` and other included `COPYING.*` notices |

Only the OpenSpiel source closure needed for the selected benchmark games,
the turn-based transformation, and retained tools is compiled. Unrelated game
implementations, Python bindings, bridge solver and full upstream test suites
are not distributed as part of this subset. Eigen headers and Abseil sources
are included to keep the C++ build independent of system library versions.

The OpenSpiel snapshot's original root license file was absent in the working
experiment tree. The standard Apache 2.0 license text is supplied alongside the
unchanged OpenSpiel files, which carry their own Apache 2.0 notices. This does
not replace or remove those notices.

Python analysis dependencies are installed separately. The optional
`open-spiel==1.4` Python package is used only by the manuscript's game-statistics
scripts and is separate from the vendored C++ experiment snapshot.
