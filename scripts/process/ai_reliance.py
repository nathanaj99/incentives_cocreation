"""
ai_reliance.py -> ai_reliance.csv: how much of each AI participant's final
submission was adopted from the AI's replies (adoption rate A_i in [0, 1]).

n-gram measures (attribution with prompt filtering):
  Each n-gram of the final text is attributed to the EARLIEST AI reply that
  contains it, extended to the longest contiguous match. If that n-gram also
  appears in the participant's own request that produced the reply, it is
  treated as human-originated (the AI echoed the participant). Matched spans
  are tiled (no double counting). A_i = attributed n-grams / all n-grams.

    adoption_3, adoption_5, adoption_10  single n
    adoption_5_10                        pooled over n = 5..10
    adoption_total                       pooled over n = 1..10

Embedding measure (adoption_embedding):
  Each sentence of the final text is attributed to the first AI reply with a
  sentence at cosine >= 0.86 (all-MiniLM-L12-v2), unless it also matches a
  sentence of the corresponding request (cosine >= 0.86 and word Jaccard
  >= 0.10). A_i = attributed words / all words.

The paper uses adoption_10 for Study 1 and adoption_3 for Study 2.

Input:  raw/submissions.csv
Output: processed/ai_reliance.csv
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import ai_submissions, output, parse_args, save_csv, should_skip, split_transcript

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


# ============================================================
# N-GRAM ATTRIBUTION
# ============================================================

def tokenize(text):
    """Lower-cased words and individual punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngram_set(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def first_reply_with_ngram(reply_tokens, n):
    """n-gram -> index of the earliest AI reply containing it."""
    first = {}
    for k, toks in enumerate(reply_tokens):
        for i in range(len(toks) - n + 1):
            first.setdefault(tuple(toks[i:i + n]), k)
    return first


def ngram_attribution(final_text, ai_texts, ai_prompts, n):
    """
    Attributed n-gram mass W_i and number of final n-grams T for one n.
    For each final n-gram start position t (left to right):
      - find the earliest reply k containing the n-gram, extend the match
        contiguously to a maximal span [t, end), and skip past it;
      - the span counts (end - t - n + 1) n-grams toward AI adoption, unless
        the starting n-gram also appears in ai_prompts[k] (human-originated).
    """
    assert len(ai_texts) == len(ai_prompts), "ai_texts and ai_prompts must have the same length."
    final = tokenize(final_text)
    T = len(final)
    if T < n:
        return {"W_i": 0, "T_ngrams": 0}

    replies = [tokenize(a) for a in ai_texts]
    prompts = [ngram_set(tokenize(p), n) for p in ai_prompts]
    first_k = first_reply_with_ngram(replies, n)

    W, t = 0, 0
    while t <= T - n:
        ng = tuple(final[t:t + n])
        if ng not in first_k:
            t += 1
            continue
        k = first_k[ng]
        reply = replies[k]
        # first occurrence in reply k, then extend the match token by token
        a = next(j for j in range(len(reply) - n + 1) if tuple(reply[j:j + n]) == ng)
        end, a_end = t + n, a + n
        while end < T and a_end < len(reply) and final[end] == reply[a_end]:
            end += 1
            a_end += 1
        if ng not in prompts[k]:
            W += end - t - n + 1
        t = end  # tile: skip past the span either way
    return {"W_i": W, "T_ngrams": T - n + 1}


def adoption_ngram(final_text, ai_texts, ai_prompts, min_n, max_n):
    """A_i pooled over n = min_n..max_n (attributed / total n-grams)."""
    results = [ngram_attribution(final_text, ai_texts, ai_prompts, n) for n in range(min_n, max_n + 1)]
    total = sum(r["T_ngrams"] for r in results)
    return sum(r["W_i"] for r in results) / total if total else 0.0  # text shorter than n


# ============================================================
# EMBEDDING ATTRIBUTION
# ============================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()] if text else []


def content_words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def cosine_sim_matrix(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


@dataclass
class EmbeddingConfig:
    tau_ai: float = 0.86          # sentence must match an AI sentence at >= tau_ai
    tau_prompt: float = 0.86      # ... and NOT match a request sentence at >= tau_prompt
    jaccard_min_prompt: float = 0.10  # lexical guard against false request matches


def adoption_embedding(final_text, ai_texts, ai_prompts, model, config=EmbeddingConfig()):
    """Share of final-text words in sentences attributed to the AI."""
    assert len(ai_texts) == len(ai_prompts), "ai_texts and ai_prompts must have the same length."
    final_spans = split_sentences(final_text)
    if not final_spans or not ai_texts:
        return 0.0

    span_mass = np.array([len(content_words(s)) for s in final_spans], dtype=float)
    total_mass = float(len(content_words(final_text))) or float(span_mass.sum()) or 1.0

    def encode(spans):
        return model.encode(spans, convert_to_numpy=True, normalize_embeddings=False) if spans else None

    final_emb = encode(final_spans)
    ai_spans = [split_sentences(a) for a in ai_texts]
    prompt_spans = [split_sentences(p) for p in ai_prompts]
    ai_embs = [encode(s) for s in ai_spans]
    prompt_embs = [encode(s) for s in prompt_spans]

    attributable = np.zeros(len(final_spans), dtype=bool)
    for u, span in enumerate(final_spans):
        f = final_emb[u:u + 1]
        # earliest reply with a sufficiently similar sentence
        k = next((k for k, e in enumerate(ai_embs)
                  if e is not None and cosine_sim_matrix(f, e).max() >= config.tau_ai), None)
        if k is None:
            continue
        if prompt_embs[k] is None:
            attributable[u] = True  # no request text to filter against
            continue
        sims = cosine_sim_matrix(f, prompt_embs[k]).reshape(-1)
        best = int(sims.argmax())
        prompt_match = (sims[best] >= config.tau_prompt
                        and jaccard(content_words(span), content_words(prompt_spans[k][best]))
                        >= config.jaccard_min_prompt)
        attributable[u] = not prompt_match

    return float((span_mass * attributable).sum()) / total_mass


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args(__doc__)
    model = None
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "ai_reliance.csv")
        if should_skip(out, args.force):
            continue
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL)

        rows = []
        ai = ai_submissions(study)
        for _, row in tqdm(ai.iterrows(), total=len(ai)):
            prompts, replies = split_transcript(row["ai_transcript"])
            prompts = prompts[:len(replies)]  # drop a trailing unanswered request
            final = row["final_text"]
            rows.append({
                "participant_id": row["participant_id"],
                "group_key": row["group_key"],
                "adoption_5_10": adoption_ngram(final, replies, prompts, 5, 10),
                "adoption_10": adoption_ngram(final, replies, prompts, 10, 10),
                "adoption_total": adoption_ngram(final, replies, prompts, 1, 10),
                "adoption_embedding": adoption_embedding(final, replies, prompts, model),
                "adoption_3": adoption_ngram(final, replies, prompts, 3, 3),
                "adoption_5": adoption_ngram(final, replies, prompts, 5, 5),
            })
        save_csv(pd.DataFrame(rows), out)


if __name__ == "__main__":
    main()
