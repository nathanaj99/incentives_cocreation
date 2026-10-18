# Figures and tables

Run any script from the repository root, e.g. `python scripts/viz/figure_2.py`.
Shared style (`palette_map`, `pretty_map`, rcParams), paths, and the jackknife /
Benjamini–Hochberg statistics live in `common.py`.

| Script | Output(s) | Content |
|---|---|---|
| `figure_1.py` | `figs/figure_1.png` | Study 1: draft vs. final similarity, trajectory, PCA* |
| `figure_2.py` | `figs/figure_2.png` | Effect-size heatmaps, both studies |
| `figure_3.py` | `figs/figure_3.png` | AI adoption and AI use |
| `figure_4.py` | `figs/figure_4.png` | Effects by Low / High adoption |
| `figure_5.py` | `figs/figure_5.png` | Technical quality and quality-conditional similarity |
| `figure_s10.py` | `figs/figure_s10.png` | Correlation of adoption measures |
| `figure_s11.py` | `figs/figure_s11_top.png`, `figs/figure_s11_bottom.png` | Brainstorm similarity; difference-in-differences |
| `figure_s12.py` | `figs/figure_s12_{top,bottom}.png` | Draft vs. final, all metrics (Study 1 / 2) |
| `figure_s13.py` | `figs/figure_s13_{top,bottom}.png` | Trajectories, all metrics (Study 1 / 2) |
| `figure_s14.py` | `figs/figure_s14_{top,bottom}.png` | PCA across embedding models (Study 1 / 2) |
| `figure_s15.py` | `figs/figure_s15.png` | Figure 1 for Study 2* |
| `figure_s16.py` | `figs/figure_s16.png` | Study 2 adoption with n = 10 |
| `table_s1_s2.py` | `tables/table_s1.tex`, `tables/table_s2.tex` | Balance tables (Study 1 / 2) |
| `table_s3_s11.py` | `tables/table_s3.tex`, `tables/table_s11.tex` | High vs. Low adopter demographics (Study 1 / 2) |
| `table_s4_s9.py` | `tables/table_s4.tex`, `tables/table_s9.tex` | Main effect sizes (Study 1 / 2) |
| `table_s5_s10.py` | `tables/table_s5.tex`, `tables/table_s10.tex` | Effects by Low / High adoption (Study 1 / 2) |
| `table_s6.py` | `tables/table_s6.tex` | LLM quality-rating reliability and validity |
| `table_s7.py` | `tables/table_s7.tex` | AI × incentive interaction |
| `table_s8.py` | `tables/table_s8.tex` | Scale-free homogenization measures |

Tables S1/S2 and S3/S11 read only the published demographic summary statistics
(`data/study*/processed/demographics_summary.csv`); individual-level
demographics are not part of this repository.

\* Re-embeds texts with `sentence-transformers/all-MiniLM-L12-v2` (downloaded on first run).
