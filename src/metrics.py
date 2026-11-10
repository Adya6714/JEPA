"""Retrieval metrics: instance and class Recall@K via cosine similarity."""

import numpy as np

from . import SEED

np.random.seed(SEED)


def _l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def retrieval(query, gallery, query_labels, gallery_labels, groups=None):
    """Compute instance and class R@1 / R@5 with cosine similarity.

    Assumes row i of ``query`` is paired with row i of ``gallery`` for
    instance-level retrieval. Class metrics count any gallery item whose
    label matches the query label as a hit.

    If ``groups`` is provided (group id per gallery row, aligned so that the
    true pair of query i is gallery i), also returns dup-aware instance
    recalls: a hit if any top-K gallery row shares the true pair's group id.
    """
    query = _l2_normalize(query)
    gallery = _l2_normalize(gallery)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    sims = query @ gallery.T  # (Nq, Ng)
    ranked = np.argsort(-sims, axis=1)

    n_query = query.shape[0]
    instance_targets = np.arange(n_query)

    def recall_at_k(k, mode):
        topk = ranked[:, :k]
        if mode == "instance":
            hits = (topk == instance_targets[:, None]).any(axis=1)
        elif mode == "instance_dup":
            groups_arr = np.asarray(groups)
            true_groups = groups_arr[instance_targets]
            topk_groups = groups_arr[topk]
            hits = (topk_groups == true_groups[:, None]).any(axis=1)
        elif mode == "class":
            topk_labels = gallery_labels[topk]
            hits = (topk_labels == query_labels[:, None]).any(axis=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return float(hits.mean())

    out = {
        "instance_R@1": recall_at_k(1, "instance"),
        "instance_R@5": recall_at_k(5, "instance"),
        "class_R@1": recall_at_k(1, "class"),
        "class_R@5": recall_at_k(5, "class"),
    }
    if groups is not None:
        out["instance_dup_R@1"] = recall_at_k(1, "instance_dup")
        out["instance_dup_R@5"] = recall_at_k(5, "instance_dup")
    return out
