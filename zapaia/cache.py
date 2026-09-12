"""Caché de features en SQLite, con clave (path, mtime, parámetros).

Motivo: una corrida sobre 9 GB tarda horas. Sin caché, cada cambio de pesos
obliga a reprocesar todo y un Ctrl-C pierde el trabajo.
"""
import json
import os
import sqlite3
import time

import numpy as np

from . import FEATURE_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, mtime REAL, size INTEGER, duration REAL,
    fver INTEGER, sr INTEGER, win REAL, hop REAL,
    status TEXT, error TEXT, sig BLOB, ts REAL
);
CREATE TABLE IF NOT EXISTS windows (
    path TEXT, widx INTEGER, start REAL, feats TEXT,
    PRIMARY KEY (path, widx)
);
CREATE INDEX IF NOT EXISTS idx_win_path ON windows(path);
"""


def connect(db_path):
    con = sqlite3.connect(db_path, timeout=60)
    con.executescript(SCHEMA)
    return con


def params_key(sr, win, hop):
    return (FEATURE_VERSION, sr, win, hop)


def is_fresh(con, path, sr, win, hop):
    """True si el archivo ya está procesado con estos mismos parámetros y mtime."""
    st = os.stat(path)
    row = con.execute(
        "SELECT mtime, fver, sr, win, hop, status FROM files WHERE path=?", (path,)
    ).fetchone()
    if not row:
        return False
    mtime, fver, s, w, h, status = row
    return (abs(mtime - st.st_mtime) < 1e-6 and (fver, s, w, h) == params_key(sr, win, hop)
            and status in ("ok", "error"))


def store(con, path, sr, win, hop, duration, wins, sig, status="ok", error=None):
    st = os.stat(path)
    blob = sig.astype(np.float32).tobytes() if sig is not None else None
    con.execute("DELETE FROM windows WHERE path=?", (path,))
    con.execute(
        "REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (path, st.st_mtime, st.st_size, duration, FEATURE_VERSION, sr, win, hop,
         status, error, blob, time.time()),
    )
    if wins:
        con.executemany(
            "REPLACE INTO windows VALUES (?,?,?,?)",
            [(path, i, s, json.dumps(f)) for i, s, f in wins],
        )
    con.commit()


def load_windows(con, sr, win, hop, prefix=None):
    """Devuelve (rows_de_ventanas, info_por_archivo) para los archivos vigentes."""
    q = ("SELECT path, duration FROM files WHERE status='ok' AND fver=? AND sr=? "
         "AND win=? AND hop=?")
    args = list(params_key(sr, win, hop))
    if prefix:
        q += " AND path LIKE ?"
        args.append(prefix + "%")
    files = {p: d for p, d in con.execute(q, args)}
    rows = []
    for path, widx, start, feats in con.execute(
        "SELECT path, widx, start, feats FROM windows"
    ):
        if path in files:
            d = json.loads(feats)
            d.update(path=path, widx=widx, start=start)
            rows.append(d)
    return rows, files


def load_signatures(con, prefix=None):
    q = "SELECT path, duration, sig FROM files WHERE status='ok' AND sig IS NOT NULL"
    args = []
    if prefix:
        q += " AND path LIKE ?"
        args.append(prefix + "%")
    out = []
    for path, dur, blob in con.execute(q, args):
        out.append((path, dur, np.frombuffer(blob, dtype=np.float32)))
    return out
