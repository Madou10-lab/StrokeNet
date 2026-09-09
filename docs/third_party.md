# Third-party components

- **Mamba**: `mamba_ssm/` contains Mamba 2.2.4 support code and the bidirectional/ball-conditioned modules used by StrokeNet. Copyright notices in the source and the [Apache 2.0 license](../licenses/mamba-LICENSE) are retained. Upstream: https://github.com/state-spaces/mamba.
- **CTR-GCN**: the dynamic channel-wise relation formulation is used in `models/layers/graph_temporal.py` and extended with graph-distance priors. The [upstream license](../licenses/CTR-GCN-LICENSE) is retained. Upstream: https://github.com/Uason-Chen/CTR-GCN.
- **Skeleton-action tooling**: the release retains the local torchlight utility code and skeleton preprocessing routines. The [license distributed with the local skeleton tooling](../licenses/skeleton-tooling-LICENSE) is included. These notices do not assign a new license to third-party code.
- **BadmintonDB and BST**: data remain distributed by their authors. Follow the links and citation instructions in the [BadmintonDB repository](https://github.com/kwban/badminton-db) and [BST repository](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer).

No blanket license for original StrokeNet code is assigned in this release preparation.
