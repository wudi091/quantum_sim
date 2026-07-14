# Third-party source provenance

The following upstream projects are vendored unchanged so the physical
simulator and comparison algorithms can be reproduced from a single clone.

## RELiQ

- Upstream: <https://github.com/meusert/RELiQ.git>
- Commit: `4312a8d2a79d91c2cf3d7749b1f2e3ecf0e46ce1`
- Local directory: `RELiQ/`
- License: preserved in `RELiQ/LICENSE`

## Q-DDCA

- Upstream: <https://github.com/QNLab-USTC/QDDCA.git>
- Commit: `c712a38d4f1ee8b56c271a0825570d50322ccb0f`
- Local directory: `QDDCA/`
- Citation and usage notes: preserved in `QDDCA/README.md`

The BatchSwap implementation composes RELiQ through an external adapter and
does not edit the vendored RELiQ or Q-DDCA source files.
