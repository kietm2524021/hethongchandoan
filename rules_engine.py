# -*- coding: utf-8 -*-
"""Luật thực nghiệm: cụm từ triệu chứng -> nhãn ICD. Chỉ khai phá từ train."""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def mine_rules(texts, y, labels, min_support=15, min_confidence=0.62,
               min_lift=1.18, max_per_label=20):
    if len(texts) < min_support or len(set(y)) < 2:
        return []
    vector = CountVectorizer(binary=True, ngram_range=(2,3),
                             token_pattern=r'(?u)\b\w+\b',
                             min_df=min_support, max_features=18000,
                             lowercase=False)
    try:
        mat = vector.fit_transform(texts)
    except ValueError:
        return []
    n = mat.shape[0]
    counts = np.asarray(mat.sum(axis=0)).ravel()
    names = vector.get_feature_names_out()
    rules = []
    y = np.asarray(y)
    for lab in labels:
        class_index = y == lab
        prior = class_index.mean()
        if prior <= 0: continue
        matched = np.asarray(mat[class_index].sum(axis=0)).ravel()
        for j, tp in enumerate(matched):
            support = int(counts[j]); tp = int(tp)
            if tp < max(3, min_support // 3): continue
            confidence = tp / support
            lift = confidence / prior
            if confidence < min_confidence or lift < min_lift: continue
            phrase = str(names[j]); tokens=phrase.split()
            if len(tokens)<2 or phrase.startswith('bệnh nhân ') and len(tokens)==2:
                continue
            strength = float(confidence * math.log1p(tp) * math.log1p(lift))
            rules.append({'phrase':phrase,'icd':str(lab),'support':support,
                          'positive':tp,'confidence':round(confidence,5),
                          'lift':round(lift,5),'strength':round(strength,5)})
    selected = []
    for lab in labels:
        candidate = sorted([r for r in rules if r['icd']==lab],
                           key=lambda r:(-r['strength'], -r['positive'],len(r['phrase'])))
        selected += candidate[:max_per_label]
    return selected


def rule_evidence(text, rules, labels):
    scores = {label:0. for label in labels}
    matched = []
    for r in rules:
        if re.search(r'(?<!\w)'+re.escape(r['phrase'])+r'(?!\w)', text):
            matched.append(r)
    # Các n-gram gần giống nhau dễ bị tính lặp, chỉ lấy 3 luật mỗi nhóm.
    for label in labels:
        matches = sorted((x for x in matched if x['icd']==label),
                         key=lambda x:-x['strength'])[:3]
        scores[label] = sum(x['strength'] for x in matches)
    return scores, sorted(matched, key=lambda r:-r['strength'])[:12]


def score_rule_only(text, rules, labels):
    scores, matched = rule_evidence(text, rules, labels)
    if not any(scores.values()):
        return None, matched
    return max(labels, key=lambda k:scores[k]), matched


def score_hybrid(text, probabilities, rules, labels, rule_weight=0.25):
    arr = np.asarray(probabilities, dtype=float)
    scores, matched = rule_evidence(text, rules, labels)
    evidence = np.array([scores[lab] for lab in labels], dtype=float)
    if evidence.sum() > 0:
        evidence = evidence/evidence.sum()
        arr = (1-rule_weight)*arr + rule_weight*evidence
    return labels[int(arr.argmax())], matched
