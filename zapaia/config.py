"""Configuración local NO versionada: bandas, IDs de carpetas de Drive, resource keys.

El repo es público. Nada de IDs, URLs de Drive ni claves en el código ni en la
documentación: todo se lee de `zapaia_local.json` (gitignored). Hay un
`zapaia_local.example.json` con la forma esperada.

Estructura por BANDA: hoy solo Nebulosa; mañana más bandas, cada una con su
carpeta de ensayos en Drive, su carpeta de salida, sus fotos y su corpus local.

    {"banda_default": "nebulosa",
     "bandas": {"nebulosa": {"nombre": "Nebulosa", "root": "nebulosa",
                             "drive": {"ensayos": {...}, "salida": {...}, "imagenes": {...}}}}}
"""
import json
import os
from pathlib import Path

_ARCHIVO = Path(os.environ.get("ZAPAIA_LOCAL", "zapaia_local.json"))
_cache = None


def cargar():
    global _cache
    if _cache is None:
        if not _ARCHIVO.exists():
            raise SystemExit(
                f"Falta {_ARCHIVO}: copiá zapaia_local.example.json a {_ARCHIVO.name} y completá "
                "los IDs de Drive (no se versionan, el repo es público).")
        _cache = json.loads(_ARCHIVO.read_text(encoding="utf-8"))
    return _cache


def banda(clave=None):
    """Config de una banda (dict con nombre, root, drive). Sin clave: la default."""
    c = cargar()
    clave = clave or os.environ.get("ZAPAIA_BANDA") or c.get("banda_default")
    bandas = c.get("bandas", {})
    if clave not in bandas:
        raise SystemExit(f"zapaia_local.json: banda '{clave}' no definida (hay: {', '.join(bandas) or 'ninguna'})")
    b = dict(bandas[clave])
    b.setdefault("clave", clave)
    b.setdefault("nombre", clave.capitalize())
    b.setdefault("root", clave)
    return b


def remote(carpeta, banda_clave=None):
    """Remote de rclone para una carpeta de Drive de la banda: 'ensayos', 'salida', 'imagenes'."""
    b = banda(banda_clave)
    d = b.get("drive", {}).get(carpeta)
    if not d or not d.get("folder_id"):
        raise SystemExit(f"zapaia_local.json: falta bandas.{b['clave']}.drive.{carpeta}.folder_id")
    r = f"gdrive,root_folder_id={d['folder_id']}"
    if d.get("resource_key"):
        r += f",resource_key={d['resource_key']}"
    return r + ":"
