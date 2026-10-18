"""
Pairwise-similarity metrics and the participant-level outcome Y_i.

For a set of texts (one group of participants), Y_i is participant i's mean
similarity to every OTHER text in the set:

    Y_i = (1 / (n_valid - 1)) * sum_{j != i, j valid} S[i, j]

Texts that are missing (NaN, e.g. a snapshot below the word minimum) get
Y_i = NaN and are excluded from everyone else's average.

Metrics (column order of every Y*.csv file):

  Embedding cosine similarity
    all-MiniLM-L12-v2, all-mpnet-base-v2, Qwen/Qwen3-Embedding-0.6B,
    google/embeddinggemma-300m, AnnaWegmann/Style-Embedding
                       -- sentence-transformers, normalized embeddings
    byt5               -- google/byt5-small encoder, mean-pooled
    star               -- AIDA-UPM/star (authorship style), pooler output
  Lexical, computed on each pair of texts
    compression_ratio  -- gzip ratio of the concatenated pair
    self_bleu          -- sentence BLEU (asymmetric: S[i, j] = BLEU(hyp=j | ref=i))
    rougel             -- ROUGE-L F1 on normalized tokens
    n_gram_diversity_score_{2,3}
                       -- unique / total n-grams, averaged over n = 1..N
                          (higher = MORE diverse, unlike every other metric)
"""

import gc
import gzip
import re
import unicodedata
import warnings

import numpy as np

EMBEDDING_MODELS = [
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
    "AnnaWegmann/Style-Embedding",
]
LEXICAL_METRICS = [
    "compression_ratio",
    "self_bleu",
    "rougel",
    "n_gram_diversity_score_2",
    "n_gram_diversity_score_3",
]
ALL_METRICS = EMBEDDING_MODELS + ["byt5", "star"] + LEXICAL_METRICS


def default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_valid(text):
    return isinstance(text, str)


def _free_memory(device):
    gc.collect()
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


# ============================================================
# LEXICAL METRICS
# ============================================================

def _normalize(text):
    """Lower-cased word tokens; punctuation dropped, contractions kept."""
    t = unicodedata.normalize("NFKC", text)
    t = (t.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"')
          .replace("—", " ").replace("–", " ")
          .replace("…", " "))
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)  # stray (non-internal) apostrophes
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()


def compression_ratio(texts):
    """gzip compression ratio (original / compressed size) of the joined texts."""
    payload = " ".join(texts).encode("utf-8")
    compressed = gzip.compress(payload)
    return round(len(payload) / len(compressed), 3)


def ngram_diversity_score(texts, num_n=4):
    """
    Average over n = 1..num_n of (# unique n-grams) / (# n-grams), pooled over
    documents without forming n-grams across document boundaries.
    """
    from nltk.util import ngrams

    total_score = 0.0
    for n in range(1, num_n + 1):
        uniq, total = set(), 0
        for doc in texts:
            tokens = _normalize(doc)
            if len(tokens) < n:
                continue
            ngs = list(ngrams(tokens, n))
            uniq.update(ngs)
            total += len(ngs)
        total_score += len(uniq) / total if total > 0 else 0.0
    return round(total_score, 3)


def _lcs_len(a, b):
    """Length of the longest common subsequence (O(|a| |b|) DP)."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = temp
    return dp[len(b)]


def rouge_l_f1(texts):
    """ROUGE-L F1 between texts[0] and texts[1] (symmetric)."""
    c_toks, r_toks = _normalize(texts[0]), _normalize(texts[1])
    if not c_toks or not r_toks:
        return 0
    lcs = _lcs_len(c_toks, r_toks)
    p, r = lcs / len(c_toks), lcs / len(r_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


_BLEU = None


def self_bleu(a, b):
    """Sentence BLEU of hypothesis b against reference a, rescaled to [0, 1]."""
    global _BLEU
    if _BLEU is None:
        from sacrebleu.metrics import BLEU
        _BLEU = BLEU(tokenize="13a", effective_order=True)
    return _BLEU.sentence_score(b, [a]).score / 100.0


def pairwise_matrix(texts, fn, symmetric=True):
    """
    S[i, j] = fn([texts[i], texts[j]])           if symmetric
            = fn(texts[i], texts[j])             otherwise (both directions)
    Pairs involving a missing text are NaN; the diagonal is unused.
    """
    n = len(texts)
    S = np.full((n, n), np.nan)
    valid = [is_valid(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if not (valid[i] and valid[j]):
                continue  # leave NaN
            if symmetric:
                S[i, j] = S[j, i] = float(fn([texts[i], texts[j]]))
            else:
                S[i, j] = float(fn(texts[i], texts[j]))
                S[j, i] = float(fn(texts[j], texts[i]))
    return S


LEXICAL_FUNCTIONS = {
    "compression_ratio": (compression_ratio, True),
    "self_bleu": (self_bleu, False),
    "rougel": (rouge_l_f1, True),
    "n_gram_diversity_score_2": (lambda t: ngram_diversity_score(t, num_n=2), True),
    "n_gram_diversity_score_3": (lambda t: ngram_diversity_score(t, num_n=3), True),
}


# ============================================================
# EMBEDDING ENCODERS
# ============================================================

class SentenceTransformerEncoder:
    def __init__(self, model_id, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, device=device)
        self.device = device

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def close(self):
        del self.model
        _free_memory(self.device)


class ByT5Encoder:
    """
    google/byt5-small encoder, mean-pooled over the (byte) token dimension
    and L2-normalized. Note: the mean includes padding positions; texts are
    length-bucketed before batching to keep padding small (as in the
    original analysis).
    """

    def __init__(self, device, model_id="google/byt5-small", batch_size=8, max_length=512):
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        # fp16 on GPU as in the original runs; fp32 elsewhere
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = T5EncoderModel.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length

    def encode(self, texts):
        import torch
        import torch.nn.functional as F

        lengths = [len(self.tok(t, truncation=True, max_length=self.max_length).input_ids) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        embs = []
        with torch.inference_mode():
            for i in range(0, len(order), self.batch_size):
                batch = [texts[k] for k in order[i:i + self.batch_size]]
                x = self.tok(batch, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
                x = {k: v.to(self.device) for k, v in x.items()}
                h = self.model(**x).last_hidden_state
                embs.append(F.normalize(h.mean(dim=1), p=2, dim=1).to("cpu", dtype=torch.float32))
        embs = torch.cat(embs, dim=0).numpy()
        return embs[np.argsort(order)]  # restore input order

    def close(self):
        del self.model, self.tok
        _free_memory(self.device)


class StarEncoder:
    """AIDA-UPM/star authorship-style embeddings (roberta-large tokenizer)."""

    def __init__(self, device, model_id="AIDA-UPM/star", tok_id="roberta-large", batch_size=16):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device, self.batch_size = device, batch_size

    def encode(self, texts):
        import torch
        import torch.nn.functional as F

        embs = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                x = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                             return_tensors="pt")
                x = {k: v.to(self.device) for k, v in x.items()}
                out = self.model(**x)
                # pooler output when available, otherwise the CLS token
                emb = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                    else out.last_hidden_state[:, 0, :]
                embs.append(F.normalize(emb, p=2, dim=1).to("cpu", dtype=torch.float32))
        return torch.cat(embs, dim=0).numpy()

    def close(self):
        del self.model, self.tok
        _free_memory(self.device)


def load_encoder(metric, device):
    if metric == "byt5":
        return ByT5Encoder(device)
    if metric == "star":
        return StarEncoder(device)
    return SentenceTransformerEncoder(metric, device)


def embedding_matrix(texts, encoder):
    """Cosine-similarity matrix; rows / columns of missing texts are NaN."""
    valid = np.array([is_valid(t) for t in texts])
    S = np.full((len(texts), len(texts)), np.nan, dtype=np.float32)
    if valid.any():
        embs = encoder.encode([t for t, v in zip(texts, valid) if v]).astype(np.float32)
        S[np.ix_(valid, valid)] = embs @ embs.T
    return S


# ============================================================
# PARTICIPANT-LEVEL OUTCOMES
# ============================================================

def y_from_matrix(S):
    """Y_i = mean over j != i of S[i, j], ignoring NaN entries."""
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (missing texts) -> NaN
        return np.nanmean(S, axis=1).tolist()


class Metric:
    """
    A similarity metric that can be applied to many groups of texts.
    Embedding models are loaded lazily on first use; call close() to free them.
    """

    def __init__(self, name, device=None):
        if name not in ALL_METRICS:
            raise ValueError(f"Unknown metric {name!r}")
        self.name = name
        self.device = device
        self.encoder = None

    def matrix(self, texts):
        if self.name in LEXICAL_FUNCTIONS:
            fn, symmetric = LEXICAL_FUNCTIONS[self.name]
            return pairwise_matrix(texts, fn, symmetric=symmetric)
        if self.encoder is None:
            self.encoder = load_encoder(self.name, self.device or default_device())
        return embedding_matrix(texts, self.encoder)

    def y(self, texts, min_valid=2):
        """
        Y_i for one group. If fewer than `min_valid` texts are present, every
        Y_i in the group is NaN.
        """
        if sum(is_valid(t) for t in texts) < min_valid:
            return [np.nan] * len(texts)
        return y_from_matrix(self.matrix(texts))

    def close(self):
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None


def y_table(groups, metrics=ALL_METRICS, device=None, min_valid=2):
    """
    groups: ordered {key: list of texts}. Returns {metric: {key: [Y_i, ...]}}.
    Each metric's model is loaded once and freed before the next metric.
    """
    out = {}
    for name in metrics:
        print(f"    {name}")
        metric = Metric(name, device)
        try:
            out[name] = {key: metric.y(texts, min_valid=min_valid) for key, texts in groups.items()}
        finally:
            metric.close()
    return out


def y_frame(df, text_col, group_order, metrics=ALL_METRICS, device=None):
    """
    Y_i for each participant within their group_key, as a DataFrame with
    columns participant_id, group_key, <metrics...>, rows ordered by group_order
    (then by input order within each group).
    """
    import pandas as pd

    groups = {g: df.loc[df["group_key"] == g, text_col].tolist() for g in group_order}
    ids = {g: df.loc[df["group_key"] == g, "participant_id"].tolist() for g in group_order}
    table = y_table(groups, metrics, device)
    frame = pd.DataFrame({
        "participant_id": sum((ids[g] for g in group_order), []),
        "group_key": sum(([g] * len(ids[g]) for g in group_order), []),
    })
    for name in metrics:
        frame[name] = sum((table[name][g] for g in group_order), [])
    return frame


def write_y(frame, path, metrics, id_cols=("participant_id", "group_key")):
    """
    Save a Y frame. When only a subset of metrics was computed and the file
    already exists, update just those columns and keep the others.
    """
    import pandas as pd
    from common import save_csv

    if set(metrics) != set(ALL_METRICS) and path.exists():
        old = pd.read_csv(path)
        if len(old) != len(frame) or not (old["participant_id"].values == frame["participant_id"].values).all():
            raise ValueError(f"{path} has different rows; rerun with all metrics")
        for m in metrics:
            old[m] = frame[m].values
        frame = old[[c for c in old.columns if c not in ALL_METRICS]
                    + [m for m in ALL_METRICS if m in old.columns]]
    save_csv(frame, path)


def add_metric_args(parser):
    """CLI options shared by the Y scripts."""
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS,
                        metavar="METRIC", help="subset of metrics to compute (default: all)")
    parser.add_argument("--device", default=None,
                        help="torch device for embedding models (default: cuda > mps > cpu)")
