"""Configuración local NO versionada: IDs de carpetas de Drive, resource keys.

El repo es público. Nada de IDs, URLs de Drive ni claves en el código ni en la
documentación: todo se lee de `zapaia_local.json` (gitignored). Hay un
`zapaia_local.example.json` con la forma esperada.
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


def remote(clave):
    """Remote de rclone para una carpeta de Drive, p. ej. 'salida', 'nebulosa', 'imagenes'.

    Formato del json: {"drive": {"salida": {"folder_id": "...", "resource_key": "..."}}}.
    """
    d = cargar().get("drive", {}).get(clave)
    if not d or not d.get("folder_id"):
        raise SystemExit(f"zapaia_local.json: falta drive.{clave}.folder_id")
    r = f"gdrive,root_folder_id={d['folder_id']}"
    if d.get("resource_key"):
        r += f",resource_key={d['resource_key']}"
    return r + ":"
