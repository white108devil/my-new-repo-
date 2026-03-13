from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Put your CSV in data/raw and "
            "update config.yaml if the filename is different."
        )
    return pd.read_csv(path)


def split_features_labels(
    df: pd.DataFrame, label_column: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    if label_column not in df.columns:
        lower_map = {col.lower(): col for col in df.columns}
        if label_column.lower() in lower_map:
            label_column = lower_map[label_column.lower()]
        else:
            raise ValueError(
                f"Label column '{label_column}' not found in dataset. "
                "Set data.label_column in config.yaml."
            )
    y = df[label_column].to_numpy()
    return df.drop(columns=[label_column]), y


def normalize_drop_columns(drop_columns: Iterable[str] | dict | str | None) -> list[str]:
    if drop_columns is None:
        return []
    if isinstance(drop_columns, dict):
        return [str(key) for key in drop_columns.keys()]
    if isinstance(drop_columns, str):
        return [drop_columns]
    normalized: list[str] = []
    for item in drop_columns:
        if isinstance(item, dict):
            normalized.extend(str(key) for key in item.keys())
        else:
            normalized.append(str(item))
    return normalized


def fill_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
        return df
    if strategy == "zero":
        return df.fillna(0)
    return df.ffill().bfill()


def frequency_encode(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    working = df.copy()
    for col in columns:
        freq = working[col].value_counts(normalize=True, dropna=True)
        working[col] = working[col].map(freq).fillna(0.0)
    return working


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_COLUMN_ALIASES = {
    "customer_id": ["customer_id", "cust_id", "customerid", "custid"],
    "customer_name": ["customer_name", "customername", "name"],
    "customer_email": ["customer_email", "email", "customeremail"],
    "gender": ["gender", "sex"],
    "age": ["age", "customer_age", "cardholder_age"],
    "state": ["state", "province"],
    "city": ["city", "town"],
    "bank_branch": ["bank_branch", "branch", "bankbranch"],
    "account_type": ["account_type", "acct_type", "accounttype"],
    "transaction_id": ["transaction_id", "txn_id", "transactionid", "txnid"],
    "transaction_date": ["transaction_date", "trans_date", "transactiondate", "date"],
    "transaction_time": ["transaction_time", "trans_time", "transactiontime", "time"],
    "transaction_amount": ["transaction_amount", "amount", "transactionamount", "amt"],
    "merchant_id": ["merchant_id", "merchantid"],
    "transaction_type": ["transaction_type", "transactiontype", "transtype", "type"],
    "merchant_category": ["merchant_category", "merchantcategory", "category"],
    "account_balance": ["account_balance", "balance", "accountbalance"],
    "transaction_device": ["transaction_device", "transactiondevice", "device"],
    "transaction_location": ["transaction_location", "transactionlocation", "location"],
    "device_type": ["device_type", "devicetype"],
    "transaction_currency": ["transaction_currency", "currency", "curr"],
    "transaction_description": ["transaction_description", "description", "txn_description"],
}


def _build_rename_map(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_normalize_col(col): col for col in columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_col(alias)
            if key in normalized:
                rename_map[normalized[key]] = canonical
                break
    return rename_map


@dataclass
class FeatureReport:
    created_features: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)


class BankFeatureEngineer:
    def __init__(self, drop_columns: Iterable[str] | dict | str | None, fillna_strategy: str):
        self.drop_columns = normalize_drop_columns(drop_columns)
        self.fillna_strategy = fillna_strategy
        self.report = FeatureReport()

    def _coerce_numeric(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if self.fillna_strategy == "median":
            return numeric.fillna(numeric.median())
        if self.fillna_strategy == "zero":
            return numeric.fillna(0)
        return numeric.ffill().bfill()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        rename_map = _build_rename_map(working.columns)
        working = working.rename(columns=rename_map)

        drop_cols = set(self.drop_columns)
        for original, canonical in rename_map.items():
            if original in drop_cols:
                drop_cols.add(canonical)

        auto_drop = {
            "customer_name",
            "customer_email",
            "transaction_description",
            "transaction_id",
            "merchant_id",
        }

        drop_cols = drop_cols.union(auto_drop)
        existing_drop = [col for col in drop_cols if col in working.columns]
        if existing_drop:
            working = working.drop(columns=existing_drop, errors="ignore")
        self.report.dropped_columns = existing_drop

        created_features: list[str] = []

        trans_dt = None
        if "transaction_date" in working.columns:
            date_series = working["transaction_date"].astype(str)
            if "transaction_time" in working.columns:
                time_series = working["transaction_time"].astype(str)
                trans_dt = pd.to_datetime(date_series + " " + time_series, errors="coerce")
                working = working.drop(columns=["transaction_time"])
            else:
                trans_dt = pd.to_datetime(date_series, errors="coerce")
            working = working.drop(columns=["transaction_date"])

        if trans_dt is not None:
            working["hour"] = trans_dt.dt.hour
            working["day"] = trans_dt.dt.day
            working["month"] = trans_dt.dt.month
            working["day_of_week"] = trans_dt.dt.weekday
            created_features.extend(["hour", "day", "month", "day_of_week"])

        if "transaction_amount" in working.columns:
            amount = self._coerce_numeric(working["transaction_amount"])
            working["transaction_amount"] = amount
            working["amount_log"] = np.log1p(amount.clip(lower=0))
            working["amount_percentile"] = amount.rank(pct=True)
            created_features.extend(["amount_log", "amount_percentile"])

            if "account_balance" in working.columns:
                balance = self._coerce_numeric(working["account_balance"])
                working["account_balance"] = balance
                working["amount_to_balance_ratio"] = amount / (balance + 1.0)
                created_features.append("amount_to_balance_ratio")

        if "customer_id" in working.columns and "transaction_amount" in working.columns:
            avg_amount = working.groupby("customer_id")["transaction_amount"].transform("mean")
            working["avg_amount"] = avg_amount
            working["amount_deviation"] = working["transaction_amount"] - avg_amount
            working["amount_deviation_ratio"] = working["transaction_amount"] / (avg_amount + 1.0)
            created_features.extend(["avg_amount", "amount_deviation", "amount_deviation_ratio"])

        if "customer_id" in working.columns and trans_dt is not None:
            working["__date_key__"] = trans_dt.dt.date
            velocity = working.groupby(["customer_id", "__date_key__"])["__date_key__"].transform(
                "size"
            )
            working["transaction_velocity"] = velocity
            working = working.drop(columns=["__date_key__"])
            created_features.append("transaction_velocity")

        if "transaction_location" in working.columns:
            loc = working["transaction_location"].astype(str).str.lower()
            match_series = pd.Series(False, index=working.index)
            if "city" in working.columns:
                city = working["city"].astype(str).str.lower()
                match_series = match_series | (loc == city)
            if "state" in working.columns:
                state = working["state"].astype(str).str.lower()
                match_series = match_series | (loc == state)
            working["location_match"] = match_series.astype(int)
            created_features.append("location_match")

        if "customer_id" in working.columns:
            working = working.drop(columns=["customer_id"])

        self.report.created_features = created_features

        working = fill_missing_values(working, self.fillna_strategy)

        categorical_cols = working.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            working = frequency_encode(working, categorical_cols)

        bool_cols = working.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            working[bool_cols] = working[bool_cols].astype(int)

        return working
