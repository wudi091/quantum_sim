# RELiQ Topology Inputs

This directory stores the real-world topology inputs used by the RELiQ paper's
TopoHub evaluation. The files are copied from TopoHub v1.5.1 (Git commit
`db1a31247ffd4ed5875d584c91fed305a3f94a1e`).

Each entry has two raw representations:

- `.json`: TopoHub node-link JSON. This is the representation consumed by
  `topohub.get()` in the official RELiQ code.
- `.gml`: the GML representation distributed by TopoHub, retained for provenance.

The paper labels are mapped to TopoHub keys as follows:

| Paper label | TopoHub key | Nodes | Edges |
| --- | --- | ---: | ---: |
| Cost 266 | `sndlib/cost266` | 37 | 57 |
| Germany | `sndlib/germany50` | 50 | 88 |
| EU | `sndlib/nobel-eu` | 28 | 41 |
| Poland (SNDlib) | `sndlib/polska` | 12 | 18 |
| US | `topozoo/NetworkUsa` | 35 | 39 |
| Finland | `topozoo/Funet` | 24 | 27 |
| Poland (Topology Zoo) | `topozoo/PionierL3` | 27 | 32 |
| UK | `topozoo/Janetbackbone` | 28 | 43 |
| Canada | `topozoo/Bellcanada` | 48 | 64 |
| York | `topozoo/York` | 23 | 24 |

Source repositories:

- RELiQ: <https://github.com/meusert/RELiQ>
- TopoHub: <https://github.com/piotrjurkiewicz/topohub/tree/v1.5.1>

The official RELiQ implementation applies additional simulation preprocessing:
it inserts intermediate quantum repeaters when a real link is longer than
`200 km`, uses `150 km` as the subdivision distance, and may add links between
intermediate repeaters. Therefore these files are the raw downloaded inputs,
not the post-processed quantum graph.

Random training and evaluation graphs in RELiQ are generated from seeds in the
official source (`src/env/constants.py`); they are not TopoHub files and are
not duplicated here.

SHA-256 checksums for the downloaded JSON files are recorded in
`manifest.json`.
