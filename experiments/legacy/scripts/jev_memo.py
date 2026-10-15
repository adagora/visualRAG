"""Versioned, bounded-age cache of raw Jev judgments; thresholds stay outside."""
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import jev
import corpus

log = logging.getLogger('pixelrag')
TTL = 86400  # aliases can move; raw decisions are never treated as timeless


def path():
    return Path(os.environ['PIXELRAG_JEV_MEMO']) if os.environ.get('PIXELRAG_JEV_MEMO') else corpus.LAYOUT.index_dir / 'jev_judgments.sqlite'


def identity(state, questions, version):
    return hashlib.sha256(json.dumps({'model': os.environ.get('TYPESAFE_MODEL', 'jev-latest'),
        'version': version, 'state': state, 'questions': questions},
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def connection():
    db = sqlite3.connect(path(), timeout=.5)
    try:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, answers TEXT, created REAL)')
            yield db
    finally:
        db.close()


def valid(answers, questions):
    try:
        for key, q in questions.items():
            answer = answers[key]
            if q['type'] == 'noul':
                jev.number(answer, 'noul', 1)
            elif q['type'] == 'choice':
                options = q['criteria']
                p = answer['probabilities']
                if answer['choice'] not in options or set(p) != set(options):
                    return False
                jev.number(answer, 'confidence', 1)
                if abs(sum(jev.number(p, key, 1) for key in options) - 1) > .001:
                    return False
            else:
                return False
        return True
    except (ValueError, KeyError, TypeError):
        return False


def evaluate(state, questions, *, version, kind, timeout=3):
    key = identity(state, questions, version)
    if os.environ.get('PIXELRAG_JEV_MEMO_DISABLED') != '1':
        try:
            with connection() as db:
                row = db.execute('SELECT answers, created FROM judgments WHERE key=?', (key,)).fetchone()
            if row and time.time() - row[1] < TTL:
                answers = json.loads(row[0])
                if valid(answers, questions):
                    return answers
        except (sqlite3.Error, OSError, ValueError, TypeError):
            log.warning('Jev memo read unavailable; evaluating fresh')
    answers = jev.evaluate(state, questions, kind=kind, timeout=timeout)
    if valid(answers, questions) and os.environ.get('PIXELRAG_JEV_MEMO_DISABLED') != '1':
        try:
            with connection() as db:
                db.execute('DELETE FROM judgments WHERE created < ?', (time.time() - TTL,))
                db.execute('INSERT OR REPLACE INTO judgments VALUES (?, ?, ?)',
                           (key, json.dumps(answers), time.time()))
        except (sqlite3.Error, OSError):
            log.warning('Jev memo write unavailable; keeping fresh judgment')
    return answers
