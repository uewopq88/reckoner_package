"""
Configurable data-preprocessing helpers.

Column processing is driven entirely by the dataset profile in config.yaml:

    numeric_columns      -> StandardScaler (applied later, on the training split)
    categorical_columns  -> one-hot encoding (single value per cell)
    multihot_columns     -> multi-hot encoding (multi-value, split on separator)
    hash_columns         -> FeatureHasher   (multi-value / high-cardinality)

Sensitive attributes:

    sensitive_exclude_columns -> ALL removed from the feature matrix (never seen
                                 by the model during training or testing)
    sensitive_label_column    -> ONE column used to derive the fairness group
                                 label (privileged / non-privileged / unknown)

Each cell of the label column is matched against a canonical ``race_list``.
Rows whose race cannot be matched receive the sentinel -1 ("unknown") and are
excluded from the fairness analysis (but still used for training / performance).
"""

import difflib

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer


# Sentinel used in the sensitive_info column for rows with no known race.
UNKNOWN_SENSITIVE = -1.0

# Lazily-initialised cache for the BERT matcher (model + list embeddings).
_BERT_CACHE = {}


def _split(value, sep):
    """Split a raw cell into a list of stripped, non-empty tokens."""
    if pd.isna(value):
        return []
    return [tok.strip() for tok in str(value).split(sep) if tok.strip()]


# ---------------------------------------------------------------------------
# Race-value matching
# ---------------------------------------------------------------------------

def _match_bert(token, race_list, threshold):
    """Match a token to the closest race tag using sentence-transformer
    embeddings and cosine similarity. Imported lazily so the dependency is
    only required when match_method='bert'."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise ImportError(
            "match_method='bert' requires the 'sentence-transformers' package. "
            "Install it (pip install sentence-transformers) or use "
            "match_method='fuzzy'."
        ) from exc

    if 'model' not in _BERT_CACHE:
        _BERT_CACHE['model'] = SentenceTransformer('all-MiniLM-L6-v2')
    model = _BERT_CACHE['model']

    key = tuple(race_list)
    if _BERT_CACHE.get('list_key') != key:
        _BERT_CACHE['list_key'] = key
        _BERT_CACHE['list_emb'] = model.encode(list(race_list),
                                               convert_to_tensor=True)
    list_emb = _BERT_CACHE['list_emb']

    from sentence_transformers import util  # noqa: F811 (re-import for scope)
    token_emb = model.encode(token, convert_to_tensor=True)
    scores = util.cos_sim(token_emb, list_emb)[0]
    best_idx = int(scores.argmax())
    if float(scores[best_idx]) >= threshold:
        return race_list[best_idx]
    return None


def match_race(token, race_list, method='fuzzy', threshold=0.6):
    """Map a single raw race token to a canonical tag from ``race_list``.

    Returns the matched tag, or None if nothing matches well enough.
    """
    if not token:
        return None

    if method == 'exact':
        return token if token in race_list else None

    if method == 'keyword':
        low = token.lower()
        for tag in race_list:
            if tag.lower() in low or low in tag.lower():
                return tag
        return None

    if method == 'fuzzy':
        matches = difflib.get_close_matches(token, race_list, n=1, cutoff=threshold)
        return matches[0] if matches else None

    if method == 'bert':
        return _match_bert(token, race_list, threshold)

    raise ValueError("Unknown match_method: {}".format(method))


def build_sensitive_info(series, race_list, privileged_race, sep=';',
                         method='fuzzy', threshold=0.6):
    """Build the fairness group label for each row.

    Rule (per project spec):
      * unknown (-1)         : no token matches any race in the list
      * non-privileged (0)   : the row contains ANY non-privileged race
      * privileged (1)       : the row's matched races are exclusively the
                               privileged race
    """
    if series is None:
        return np.full(len(series) if series is not None else 0,
                       UNKNOWN_SENSITIVE)

    token_cache = {}
    labels = []
    for raw in series:
        matched = set()
        for tok in _split(raw, sep):
            if tok not in token_cache:
                token_cache[tok] = match_race(tok, race_list, method, threshold)
            tag = token_cache[tok]
            if tag is not None:
                matched.add(tag)

        if not matched:
            labels.append(UNKNOWN_SENSITIVE)
        elif any(tag != privileged_race for tag in matched):
            labels.append(0.0)
        else:
            labels.append(1.0)

    return np.array(labels, dtype=float)


# ---------------------------------------------------------------------------
# Feature encoders
# ---------------------------------------------------------------------------

def multihot_encode(series, sep=';'):
    """Multi-hot encode a multi-value categorical column (split on ``sep``)."""
    lists = series.apply(lambda v: _split(v, sep))
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(lists)
    cols = ["{}_{}".format(series.name, cls) for cls in mlb.classes_]
    return pd.DataFrame(arr, columns=cols, index=series.index).astype(float)


def hash_encode(series, n_features=20, sep=';', prefix=None):
    """FeatureHasher-encode a (multi-value / high-cardinality) column.

    The cell is split on ``sep`` into tokens first, so a value such as
    "Assault;Domestic Violence" contributes both tokens rather than being
    treated as one opaque string.
    """
    prefix = prefix or series.name
    tokens = series.apply(lambda v: _split(v, sep))
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    arr = hasher.transform(tokens).toarray()
    cols = ["{}_hash_{}".format(prefix, i + 1) for i in range(n_features)]
    return pd.DataFrame(arr, columns=cols, index=series.index).astype(float)


def process_features(df, numeric_columns=None, categorical_columns=None,
                     multihot_columns=None, hash_columns=None,
                     hash_n_features=None, sep=';'):
    """Assemble the feature matrix from the per-type column lists.

    Returns (X, numeric_out_columns) where numeric_out_columns is the list of
    numeric feature names present in X (used later for standardisation).
    """
    numeric_columns = list(numeric_columns or [])
    categorical_columns = list(categorical_columns or [])
    multihot_columns = list(multihot_columns or [])
    hash_columns = list(hash_columns or [])
    hash_n_features = hash_n_features or {}

    frames = []
    numeric_out = []

    if numeric_columns:
        num_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce').astype(float)
        frames.append(num_df)
        numeric_out = list(numeric_columns)

    if categorical_columns:
        cat_df = pd.get_dummies(df[categorical_columns].astype(str)).astype(float)
        frames.append(cat_df)

    for col in multihot_columns:
        frames.append(multihot_encode(df[col], sep))

    for col in hash_columns:
        n = hash_n_features.get(col, 20)
        frames.append(hash_encode(df[col], n, sep, prefix=col))

    if not frames:
        raise ValueError("No feature columns were configured.")

    X = pd.concat(frames, axis=1)
    return X, numeric_out
