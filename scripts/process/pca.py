"""
PCA.pkl: 2-D PCA of the embeddings of final submissions and AI drafts, for
each embedding model (Fig. S14).

Texts, in this order (must match scripts/viz/common.build_text_corpus):
final submissions, then first drafts, then third drafts (drafts_cleaned.csv,
non-missing only). For Study 2, every word containing "snap" (the product
name) is removed first so it does not dominate the embedding geometry.

Output: processed/PCA.pkl -- {model name: [[pc1, pc2], ...]}
"""

import pickle
import re

import pandas as pd
from sklearn.decomposition import PCA

from common import load_submissions, output, parse_args, should_skip
from similarity import EMBEDDING_MODELS, default_device, load_encoder

PCA_MODELS = EMBEDDING_MODELS + ["byt5", "star"]
STRIP_PRODUCT_NAME = {1: False, 2: True}


def text_corpus(study):
    texts = load_submissions(study)["final_text"].tolist()
    drafts = pd.read_csv(output(study, "drafts_cleaned.csv"))
    texts += drafts["first_draft"].dropna().tolist()
    texts += drafts["third_draft"].dropna().tolist()
    if STRIP_PRODUCT_NAME[study]:
        texts = [re.sub(r"\s+", " ", re.sub(r"\b\w*snap\w*\b", "", t, flags=re.IGNORECASE)).strip()
                 for t in texts]
    return texts


def main():
    def extra(parser):
        parser.add_argument("--device", default=None,
                            help="torch device (default: cuda > mps > cpu)")

    args = parse_args(__doc__, extra=extra)
    device = args.device or default_device()
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "PCA.pkl")
        if should_skip(out, args.force):
            continue
        texts = text_corpus(study)
        results = {}
        for name in PCA_MODELS:
            print(f"    {name}")
            encoder = load_encoder(name, device)
            try:
                embs = encoder.encode(texts)
            finally:
                encoder.close()
            results[name] = PCA(n_components=2, random_state=0).fit_transform(embs).tolist()
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved {out.name}")


if __name__ == "__main__":
    main()
