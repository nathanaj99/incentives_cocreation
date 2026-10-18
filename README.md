# Incentives shape how humans co-create with generative AI

Replication code for

> Nathanael Jo and Manish Raghavan. **Incentives shape how humans co-create with generative AI.** arXiv:2604.03529, 2026. [[paper]](https://arxiv.org/abs/2604.03529)

We run two pre-registered randomized experiments (Study 1: short stories, n = 200;
Study 2: product pitches, n = 400) in which participants write with or without an
AI assistant, under an incentive for either originality or technical quality.
This repository turns the experimental data into every figure and table in the paper.

## Repository structure

```
incentives_cocreation/
├── README.md
├── requirements.txt
├── run_all.sh              # runs the whole pipeline
├── data/                   # not in this repo -- download from Hugging Face (below)
│   ├── study1/
│   │   ├── raw/            # submissions.csv, snapshots.csv
│   │   └── processed/      # outputs of scripts/process/
│   └── study2/             # same layout
├── scripts/
│   ├── process/            # raw data -> data/study*/processed/
│   └── viz/                # processed data -> figs/ and tables/
├── figs/                   # figure_1.png ... figure_s16.png
└── tables/                 # table_s1.tex ... table_s11.tex
```

Each `scripts/` folder has its own README describing every script and its
outputs.

## Setup

Python 3.11:

```bash
pip install -r requirements.txt
```

## Data

The data are hosted on Hugging Face at
[nathanaeljo/ai-cocreation-incentives](https://huggingface.co/datasets/nathanaeljo/ai-cocreation-incentives),
with the same layout as `data/` above. Download them into `data/` at the root
of this repository:

```bash
huggingface-cli download nathanaeljo/ai-cocreation-incentives --repo-type dataset --local-dir data
```

The download includes both the raw data and every processed file, so the figures
and tables can be reproduced without rerunning the expensive processing steps.

To protect participants, participants are identified only by random study IDs,
snapshot timestamps are relative to each participant's first snapshot, and
demographics are released only as summary statistics. See
[`scripts/process/README.md`](scripts/process/README.md) for details.

## Reproducing the results

```bash
./run_all.sh
```

By default this reruns the cheap processing steps and regenerates all figures
and tables from the downloaded processed data. The expensive steps can be
switched on with environment variables:

| Variable | Effect |
|---|---|
| `RUN_LLM=1` | Rerun the OpenAI steps (labeling AI requests, extracting drafts, grading quality). Requires `OPENAI_API_KEY`. |
| `RUN_EMBED=1` | Recompute the similarity outcomes and PCA with all embedding models (GPU recommended). |
| `STUDIES=1` or `STUDIES=2` | Process one study only. |
| `FORCE=1` | Overwrite existing processed files. |

For example, `RUN_EMBED=1 FORCE=1 ./run_all.sh` recomputes every similarity outcome.
Individual scripts can also be run on their own from the repository root, e.g.
`python scripts/viz/figure_2.py` or `python scripts/process/compute_Y.py --study 2`.

LLM outputs are not deterministic, so rerunning `RUN_LLM=1` will not
reproduce the released labels, drafts, and quality scores exactly; the released
files are the ones used in the paper.

## Citation

```bibtex
@article{jo2026incentives,
  title   = {Incentives shape how humans co-create with generative AI},
  author  = {Jo, Nathanael and Raghavan, Manish},
  journal = {arXiv preprint arXiv:2604.03529},
  year    = {2026}
}
```

## License

Code is released under the Apache License 2.0 (see [`LICENSE`](LICENSE)).
