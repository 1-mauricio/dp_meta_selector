"""Additive UCI public-schema adapter for the frozen external amendment.

Only source-declared Feature and Target roles are used. No imputation, scaling,
class regrouping or feature selection is performed here. PublicSchema supplies
exactly the OpenML encoding, including missing buckets and fixed tanh.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .openml_benchmark import DatasetCandidate, _get_openml_data


def prepare_uci(metadata: dict, frame: pd.DataFrame, destination: Path) -> dict:
    if 'Classification' not in (metadata.get('tasks') or []):
        raise ValueError('not_declared_classification')
    variables = metadata.get('variables') or []
    targets = [str(v['name']) for v in variables if str(v.get('role','')).casefold() == 'target']
    declared = metadata.get('target_col') or []
    if isinstance(declared, str): declared = [declared]
    if len(targets) != 1 or list(map(str, declared)) != targets:
        raise ValueError('ambiguous_or_missing_declared_target')
    target = targets[0]
    features = [v for v in variables if str(v.get('role','')).casefold() == 'feature']
    names = [str(v['name']) for v in features]
    if not names or len(set(names + [target])) != len(names)+1:
        raise ValueError('invalid_feature_names')
    if frame.columns.duplicated().any() or any(n not in frame for n in names+[target]):
        raise ValueError('source_columns_disagree_with_declared_schema')
    if metadata.get('num_instances') is not None and len(frame) != int(metadata['num_instances']):
        raise ValueError('source_row_count_disagrees_with_metadata')
    # Sole permitted row cleaning: remove absent targets, before freezing n.
    clean = frame.loc[frame[target].notna(),names+[target]].copy()
    removed = len(frame)-len(clean)
    if not 200 <= len(clean) <= 50000: raise ValueError('row_count_outside_200_50000')
    kinds, domains = [], []
    for variable,name in zip(features,names):
        kind = str(variable.get('type') or '').casefold().strip()
        if kind in {'categorical','binary','nominal','ordinal'}:
            encoded_kind='nominal'
        elif kind in {'integer','real','continuous','numeric'}:
            encoded_kind='numeric'
        elif not kind:
            encoded_kind='numeric' if pd.api.types.is_numeric_dtype(clean[name]) else 'nominal'
        else: raise ValueError(f'unsupported_declared_type:{name}:{kind}')
        kinds.append(encoded_kind)
        if encoded_kind == 'nominal':
            clean[name] = clean[name].map(lambda v: str(v) if pd.notna(v) else None)
            domains.append(tuple(sorted(clean[name].dropna().unique())))
        else: domains.append(())
    clean[target] = clean[target].map(str)
    classes=tuple(sorted(clean[target].unique()))
    if not 2 <= len(classes) <= 30: raise ValueError('classes_outside_2_30')
    candidate=DatasetCandidate(1000000+int(metadata['uci_id']),None,target,str(metadata['name']),1,
        len(clean),tuple(names),tuple(kinds),tuple(domains),classes)
    if candidate.schema.encoded_dim > 512: raise ValueError('encoded_features_above_512')
    x=candidate.schema.encode(clean[names],numeric_transform='tanh')
    candidate.schema.class_indices(clean[target].to_numpy(dtype=object))
    if not np.isfinite(x).all(): raise ValueError('nonfinite_encoding')
    destination.parent.mkdir(parents=True,exist_ok=True)
    clean.to_parquet(destination,index=False)
    row=candidate.manifest_row('UCI-public-API-external-final')
    row.update(uci_id=int(metadata['uci_id']),local_data_path=str(destination),
        local_data_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        removed_missing_targets=removed,source_rows=len(frame),
        declared_origin_url=metadata.get('repository_url',''),
        declared_origin_doi=metadata.get('dataset_doi',''),
        source_data_url=metadata.get('data_url',''),dataset_id_rule='1000000 + uci_id')
    loaded,labels,schema=_get_openml_data(None,pd.Series(row),numeric_transform='tanh')
    if not np.array_equal(x,loaded): raise ValueError('local_OpenML_encoding_mismatch')
    return row
