"""Ingesta desde Google Drive con rclone, por rango de fechas.

Drive es SOURCE, no parte del algoritmo: el core sigue trabajando 100% con
archivos locales. Esto solo baja lo nuevo a `nebulosa/drive/AAAA-MM/` y deja un
manifest (`drive_files`) con id, tamaño y fecha de Drive, que es la mejor fecha
de grabación disponible (los mtime locales del corpus viejo son de copias).

Ver docs/review-2026-09-12.md §6 P7.
"""
import datetime as dt
import json
import os
import subprocess
import time

# Por ID + resource_key, no por nombre: Drive admite nombres duplicados y las
# rutas por nombre ya fallaron dos veces (carpeta duplicada; listado vacío).
# El ID vive en zapaia_local.json (gitignored): el repo es público.
def origen_default():
    from . import config
    return config.remote("ensayos")


ORIGEN_DEFAULT = None   # se resuelve con origen_default() al usarse
# Subcarpetas que NO son ensayos crudos: mezclas, proyectos y nuestras propias salidas.
EXCLUIR_DEFAULT = ("Seleccion IA - compilados", "Canciones", "REAPER", "NINJAMsessions")


def _rclone(args, timeout=600):
    r = subprocess.run(["rclone", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])}: {r.stderr.strip()[:300]}")
    return r.stdout


def listar(origen=None):
    """Todos los MP3 bajo `origen`, con Path relativo, Size, ModTime (UTC) e ID."""
    origen = origen or origen_default()
    out = _rclone(["lsjson", origen, "-R", "--files-only", "--include", "*.mp3"])
    items = json.loads(out)
    for it in items:
        it["fecha"] = dt.datetime.fromisoformat(it["ModTime"].replace("Z", "+00:00"))
    return items


def filtrar(items, desde, excluir=EXCLUIR_DEFAULT):
    """Por fecha y por subcarpeta excluida (prefijo del Path relativo)."""
    out = []
    for it in items:
        if it["fecha"] < desde:
            continue
        if any(it["Path"].startswith(ex.rstrip("/") + "/") or it["Path"] == ex for ex in excluir):
            continue
        out.append(it)
    return out


def locales_por_nombre(root):
    """{(nombre, tamaño): ruta} de todos los MP3 locales, para no bajar lo que ya está."""
    out = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".mp3"):
                p = os.path.abspath(os.path.join(r, f))
                out[(f, os.path.getsize(p))] = p
    return out


def ya_en_manifest(con, drive_id, size, modtime):
    row = con.execute("SELECT size, modtime, local_path FROM drive_files WHERE drive_id=?",
                      (drive_id,)).fetchone()
    return bool(row and row[0] == size and row[1] == modtime and os.path.exists(row[2]))


def registrar(con, it, local_path, origen):
    con.execute(
        "REPLACE INTO drive_files (drive_id, origen, path_drive, name, size, modtime, "
        "local_path, ts) VALUES (?,?,?,?,?,?,?,?)",
        (it["ID"], origen, it["Path"], it["Name"], it["Size"], it["ModTime"], local_path,
         time.time()))
    con.commit()


def bajar(it, origen, destino_root):
    """Baja un archivo a destino_root/AAAA-MM/nombre. rclone preserva el modtime."""
    mes = it["fecha"].strftime("%Y-%m")
    dest_dir = os.path.join(destino_root, mes)
    os.makedirs(dest_dir, exist_ok=True)
    # Ruta ABSOLUTA: el caché de features indexa por ruta absoluta y el manifest
    # tiene que coincidir para que fechas_locales() encuentre la fecha de Drive.
    local = os.path.abspath(os.path.join(dest_dir, it["Name"]))
    _rclone(["copyto", f"{origen}/{it['Path']}", local], timeout=1800)
    return local


def sincronizar(con, root, destino_root, desde, origen=None,
                excluir=EXCLUIR_DEFAULT, dry_run=False, log=print):
    """Devuelve (bajados, saltados_locales, saltados_manifest, errores)."""
    origen = origen or origen_default()
    items = filtrar(listar(origen), desde, excluir)
    locales = locales_por_nombre(root)
    bajados, sl, sm, errores = [], 0, 0, []
    log(f"{len(items)} MP3 en Drive desde {desde.date()} (excluyendo {', '.join(excluir)})")
    for it in sorted(items, key=lambda x: x["fecha"]):
        if ya_en_manifest(con, it["ID"], it["Size"], it["ModTime"]):
            sm += 1
            continue
        clave = (it["Name"], it["Size"])
        if clave in locales:
            # Ya lo teníamos (el corpus viejo se bajó a mano): registrar, no bajar.
            registrar(con, it, locales[clave], origen)
            sl += 1
            continue
        log(f"  {'[dry-run] ' if dry_run else ''}{it['fecha'].date()}  {it['Path']}  ({it['Size']/1e6:.1f} MB)")
        if dry_run:
            bajados.append(it["Path"])
            continue
        try:
            local = bajar(it, origen, destino_root)
            registrar(con, it, local, origen)
            bajados.append(local)
        except Exception as e:
            errores.append((it["Path"], str(e)[:200]))
            log(f"    ERROR: {e}")
    return bajados, sl, sm, errores


# ---------------------------------------------------------------------------
# Fechas de los archivos locales (para filtrar rank / compilado por rango)
# ---------------------------------------------------------------------------

def fechas_locales(con, paths):
    """{ruta: datetime} usando el manifest de Drive si existe, si no el mtime local."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(drive_files)")]
    q = "SELECT local_path, modtime" + (", btime" if "btime" in cols else ", NULL") + " FROM drive_files"
    # Fecha de CREACIÓN en Drive si existe (btime), si no la de modificación.
    manif = {lp: (bt or mt) for lp, mt, bt in con.execute(q)}
    out = {}
    for p in paths:
        if p in manif:
            out[p] = dt.datetime.fromisoformat(manif[p].replace("Z", "+00:00"))
        else:
            try:
                out[p] = dt.datetime.fromtimestamp(os.path.getmtime(p), tz=dt.timezone.utc)
            except OSError:
                pass
    return out


def desde_argumentos(meses=None, desde=None):
    """Convierte --meses / --desde en un datetime UTC."""
    if desde:
        return dt.datetime.fromisoformat(desde).replace(tzinfo=dt.timezone.utc)
    if meses:
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30 * meses)
    return None


def dia_local(fecha):
    """Día calendario en la zona horaria de la máquina (Drive guarda UTC; una
    subida a las 23:30 de acá no debe caer en el día siguiente)."""
    if hasattr(fecha, "to_pydatetime"):      # pandas Timestamp: astimezone() exige tz
        fecha = fecha.to_pydatetime()
    return fecha.astimezone().date()


def filtrar_por_fecha(d, fechas, desde=None, ultima_sesion=False):
    """Filtra el DataFrame de archivos (columna Ruta) por fecha.

    ultima_sesion: todos los archivos del DÍA más reciente con subidas de MP3.
    "La última zapada" son los que se subieron ese día, sin importar la hora.
    """
    d = d.copy()
    d["_fecha"] = d["Ruta"].map(fechas)
    d = d[d["_fecha"].notna()]
    if desde is not None:
        d = d[d["_fecha"] >= desde]
    if ultima_sesion and len(d):
        # "Última zapada" = el archivo más reciente y todos los que se subieron
        # en cadena con él: cada uno a menos de `gap_h` horas del anterior.
        # Agrupar por día calendario partía sesiones: una subida a las 02:00 UTC
        # es la noche anterior en hora local y quedaba en otro "día".
        gap_h = 12
        fechas_ord = sorted(d["_fecha"].map(lambda f: f.to_pydatetime() if hasattr(f, "to_pydatetime") else f),
                            reverse=True)
        corte = fechas_ord[0]
        for f in fechas_ord[1:]:
            if (corte - f).total_seconds() > gap_h * 3600:
                break
            corte = f
        d = d[d["_fecha"] >= corte]
    return d


def etiqueta_filtro(d, desde=None, ultima_sesion=False, meses=None):
    """Código corto para el nombre del archivo de salida: qué rango se usó."""
    if ultima_sesion and len(d):
        return "sesion-" + dia_local(d["_fecha"].max()).isoformat()
    if desde is not None and len(d):
        # Por MES, no por día: con la fecha exacta el nombre cambiaba cada día y
        # Drive se llenaba de compilados casi iguales con nombres distintos.
        if meses:
            return f"{meses}meses-hasta-{dia_local(d['_fecha'].max()).strftime('%Y-%m')}"
        return f"desde-{desde.date().isoformat()}"
    return "todo"
