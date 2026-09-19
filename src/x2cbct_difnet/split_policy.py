from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable


PATIENT_LEVEL_POLICY = "patient_id_shuffle_train_test"
CASE_MIXED_POLICY = "case_shuffle_train_test"


@dataclass(frozen=True)
class TrainTestSplit:
    train: list[Any]
    test: list[Any]
    metadata: dict[str, Any]


def normalize_split_policy(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"patient", "patient_id", "patient_level", PATIENT_LEVEL_POLICY}:
        return PATIENT_LEVEL_POLICY
    if token in {"case", "case_level", "mixed", "case_mixed", CASE_MIXED_POLICY}:
        return CASE_MIXED_POLICY
    raise ValueError(f"unsupported split policy: {value!r}")


def _item_value(item: Any, keys: tuple[str, ...]) -> str:
    for key in keys:
        if isinstance(item, dict) and item.get(key) is not None:
            return str(item[key])
        if not isinstance(item, dict) and hasattr(item, key):
            return str(getattr(item, key))
    raise KeyError(f"item is missing all keys: {keys}")


def _patient_id(item: Any) -> str:
    return _item_value(item, ("patient_id", "patient", "seriesuid"))


def _case_id(item: Any) -> str:
    return _item_value(item, ("case_id", "seriesuid", "sample_id"))


def _case_key(item: Any) -> str:
    return f"{_patient_id(item)}/{_case_id(item)}"


def _train_count(total: int, train_ratio: float) -> int:
    if not 0 < float(train_ratio) < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if total <= 0:
        return 0
    count = int(round(total * float(train_ratio)))
    if total == 1:
        return count
    return min(max(count, 1), total - 1)


def summarize_train_test_split(
    train: Iterable[Any],
    test: Iterable[Any],
    *,
    policy: str,
    train_ratio: float,
    seed: int,
) -> dict[str, Any]:
    normalized = normalize_split_policy(policy)
    train_rows = list(train)
    test_rows = list(test)
    train_patients = sorted({_patient_id(item) for item in train_rows})
    test_patients = sorted({_patient_id(item) for item in test_rows})
    train_cases = {_case_key(item) for item in train_rows}
    test_cases = {_case_key(item) for item in test_rows}
    patient_overlap = sorted(set(train_patients) & set(test_patients))
    case_overlap = sorted(train_cases & test_cases)
    return {
        "policy": normalized,
        "split_unit": "case" if normalized == CASE_MIXED_POLICY else "patient_id",
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "train_patient_count": len(train_patients),
        "test_patient_count": len(test_patients),
        "train_case_count": len(train_rows),
        "test_case_count": len(test_rows),
        "patient_overlap": patient_overlap,
        "case_overlap_count": len(case_overlap),
        "case_overlap": case_overlap,
        "allow_patient_overlap": normalized == CASE_MIXED_POLICY,
    }


def assign_train_test_split(
    rows: Iterable[Any],
    *,
    policy: str,
    train_ratio: float,
    seed: int,
) -> TrainTestSplit:
    normalized = normalize_split_policy(policy)
    row_list = list(rows)
    rng = random.Random(int(seed))
    if normalized == PATIENT_LEVEL_POLICY:
        patients = sorted({_patient_id(item) for item in row_list})
        shuffled = list(patients)
        rng.shuffle(shuffled)
        train_patients = set(shuffled[: _train_count(len(shuffled), train_ratio)])
        train = [item for item in row_list if _patient_id(item) in train_patients]
        test = [item for item in row_list if _patient_id(item) not in train_patients]
    else:
        indices = list(range(len(row_list)))
        rng.shuffle(indices)
        train_indices = set(indices[: _train_count(len(indices), train_ratio)])
        train = [item for index, item in enumerate(row_list) if index in train_indices]
        test = [item for index, item in enumerate(row_list) if index not in train_indices]
    metadata = summarize_train_test_split(train, test, policy=normalized, train_ratio=train_ratio, seed=seed)
    return TrainTestSplit(train=train, test=test, metadata=metadata)
