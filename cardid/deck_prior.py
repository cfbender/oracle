"""Deck-list prior: how much should the webcam table favour candidates that are in the clicked
board owner's linked deck list?

The app re-ranks the bundle's top five (`assets/react/src/features/webcam-table/deck-hint.ts`):
every candidate whose name is in the list gets `prior` added to its similarity, then the usual
clear-answer rule (`CLEAR_MARGIN` in `card-suggestions.tsx`) decides between recording the card
and asking. This replays that rule over held-out queries in two scenarios:

- in list: the clicked card is in the owner's list, the rest of the list is random gallery names;
- off list: the card is not in the list (a stolen or copied card, an outdated list) but its
  strongest wrong candidate is, which is exactly when the prior can do harm.

Clicks mostly land on the owner's own cards, so the summary mixes the scenarios with
`off_list_share` of clicks off-list. Needs only numpy; `cardid.evaluate --deck-prior` feeds it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

APP_CLEAR_MARGIN = 0.08  # `CLEAR_MARGIN` in card-suggestions.tsx
DEFAULT_PRIORS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


def rerank(sims: np.ndarray, in_deck: np.ndarray, prior: float) -> tuple[np.ndarray, np.ndarray]:
    """`applyDeckHint` for one query: (candidate order, hinted scores in that order). Ties keep
    the recognizer's order, as the stable `Array.prototype.sort` does."""
    scores = sims + prior * in_deck
    order = np.argsort(-scores, kind="stable")
    return order, scores[order]


def _deck_flags(names: list[str], listed: str | None, excluded: str | None, filler_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Which of one query's candidate names are in its simulated list: `listed` always is,
    `excluded` never is, and any other name is one of the random filler cards at `filler_rate`
    (drawn once per distinct name, so two arts of one card agree)."""
    drawn: dict[str, bool] = {}
    flags = []
    for name in names:
        if name == listed:
            flags.append(True)
        elif name == excluded:
            flags.append(False)
        else:
            drawn.setdefault(name, bool(rng.random() < filler_rate))
            flags.append(drawn[name])
    return np.array(flags, dtype=float)


def sweep(
    idx: np.ndarray,
    sims: np.ndarray,
    targets: np.ndarray,
    names: Sequence[str],
    priors: Sequence[float] = DEFAULT_PRIORS,
    deck_size: int = 100,
    off_list_share: float = 0.1,
    clear_margin: float = APP_CLEAR_MARGIN,
    seed: int = 0,
) -> list[dict]:
    """One row per prior. `idx`/`sims` are the top-k gallery indices and scores per query (best
    first, as the bundle's search returns them), `targets` the true gallery index, `names` the
    card name per gallery index. Correctness is by name: the board records a card by name, and
    two printings of one card are not a mistake the list could fix."""
    names = [name.lower() for name in names]
    unique_names = len(set(names))
    filler_rate = (deck_size - 1) / max(unique_names - 1, 1)
    rows = []
    for prior in priors:
        rng = np.random.default_rng(seed)  # the same simulated lists for every prior
        outcome = {"in_list": [], "off_list": []}
        for i in range(len(idx)):
            truth = names[targets[i]]
            candidates = [names[j] for j in idx[i]]
            lookalike = next((name for name in candidates if name != truth), None)
            scenarios = {
                "in_list": _deck_flags(candidates, truth, None, filler_rate, rng),
                "off_list": _deck_flags(candidates, lookalike, truth, filler_rate, rng),
            }
            for scenario, in_deck in scenarios.items():
                order, scores = rerank(sims[i], in_deck, prior)
                correct = candidates[order[0]] == truth
                clear = len(scores) > 1 and scores[0] - scores[1] >= clear_margin
                outcome[scenario].append((correct, clear))
        row: dict = {"prior": prior}
        for scenario, results in outcome.items():
            row[scenario] = _summary(np.array(results, dtype=bool).reshape(-1, 2), 1.0)
        mixed = np.concatenate([np.array(outcome["in_list"], dtype=bool), np.array(outcome["off_list"], dtype=bool)])
        weights = np.concatenate([np.full(len(idx), 1 - off_list_share), np.full(len(idx), off_list_share)])
        row["mixed"] = _summary(mixed.reshape(-1, 2), weights)
        rows.append(row)
    return rows


def _summary(results: np.ndarray, weights: np.ndarray | float) -> dict:
    """Weighted top-1, share of clicks recorded without asking, how often those are right, and
    the share of clicks silently recorded as the wrong card."""
    correct, clear = results[:, 0], results[:, 1]
    w = np.broadcast_to(np.asarray(weights, dtype=float), correct.shape)
    total = float(w.sum())
    auto = float((w * clear).sum())
    return {
        "top1": float((w * correct).sum()) / total if total else 0.0,
        "auto_rate": auto / total if total else 0.0,
        "auto_precision": float((w * (correct & clear)).sum()) / auto if auto else None,
        "silent_wrong": float((w * (clear & ~correct)).sum()) / total if total else 0.0,
    }


def suggest(rows: list[dict], precision_tolerance: float = 0.005, off_list_tolerance: float = 0.01) -> dict:
    """The prior with the best mixed top-1 that (a) keeps mixed auto-record precision within
    `precision_tolerance` of no prior and (b) silently records at most `off_list_tolerance` more
    off-list clicks as the wrong card. In-list gains must not hide what the prior costs when the
    card on the table is not in the list."""
    base = next(row for row in rows if row["prior"] == 0.0)
    safe = [
        row
        for row in rows
        if (row["mixed"]["auto_precision"] or 0.0) >= (base["mixed"]["auto_precision"] or 0.0) - precision_tolerance
        and row["off_list"]["silent_wrong"] <= base["off_list"]["silent_wrong"] + off_list_tolerance
    ]
    return max(safe, key=lambda row: (row["mixed"]["top1"], -row["prior"]))


def print_sweep(rows: list[dict], off_list_share: float) -> None:
    def pct(value: float | None) -> str:
        return "    — " if value is None else f"{value * 100:5.1f}%"

    print(f"deck-list prior (clear margin {APP_CLEAR_MARGIN}; mixed = {off_list_share:.0%} of clicks off-list)")
    print("  top1 = right card first; auto = recorded without asking; wrong = recorded without asking and wrong")
    print("  prior |  in list: top1   auto  wrong |  off list: top1   auto  wrong |  mixed: top1   auto  wrong  auto-prec")
    for row in rows:
        cells = [f"{pct(row[s]['top1'])} {pct(row[s]['auto_rate'])} {pct(row[s]['silent_wrong'])}" for s in ("in_list", "off_list", "mixed")]
        print(f"  {row['prior']:.2f}  |       {cells[0]} |        {cells[1]} |     {cells[2]}     {pct(row['mixed']['auto_precision'])}")
    best = suggest(rows)
    print(
        f"suggested prior: {best['prior']:.2f} (best mixed top-1 keeping auto-record precision within 0.5 pt of no prior "
        "and off-list silent mistakes within 1 pt)"
    )
