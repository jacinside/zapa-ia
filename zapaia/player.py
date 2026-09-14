"""Player compartible para votar pares A/B desde el celular, sin servidor.

Genera una página HTML estática con los pares de un lote. Los clips se sirven
desde Google Drive (link "cualquiera con el enlace"); los votos se guardan en
el navegador y al final se envían como texto por WhatsApp — exactamente el
formato que `zapaia feedback importar` entiende. Cada persona manda el suyo.

Por qué así y no con base de datos: un artifact de Claude con `db` es
"organization-internal" (solo miembros con sesión), y Drive no puede servir
audio dentro de un artifact. Una página estática + Drive + WhatsApp funciona
para cualquiera con el link.
"""
import html
import json
import re


def drive_url(file_id):
    # NO usar drive.google.com/uc?export=download: redirige con tipo
    # application/binary + nosniff y Chrome lo bloquea por CORB en <audio>.
    # Este endpoint entrega el MP3 directo, con CORS abierto y rangos (seek).
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"


def leer_lote_txt(path):
    """Parsea lote.txt -> [(num, nombre_A, nombre_B)]."""
    out = []
    for l in open(path, encoding="utf-8"):
        m = re.match(r"^(\d{3})\s+A: (.*?) \(\d+:\d\d\)\s+B: (.*?) \(\d+:\d\d\)$", l.strip())
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def generar(lote, pares, ids, titulo="Nebulosa · ¿cuál rescatarías?"):
    """pares: [(num, nombreA, nombreB)]; ids: {nombre_clip: drive_file_id}."""
    datos = []
    for num, na, nb in pares:
        ka, kb = f"par_{num:03d}_A.mp3", f"par_{num:03d}_B.mp3"
        if ka not in ids or kb not in ids:
            continue
        datos.append({"n": num, "a": na.replace(".mp3", ""), "b": nb.replace(".mp3", ""),
                      "ua": drive_url(ids[ka]), "ub": drive_url(ids[kb])})
    datos_json = json.dumps(datos, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(titulo)} · lote {lote}</title>
<style>
:root{{--bg:#F3F4F1;--ink:#1C1F1D;--ink2:#4E5652;--line:#CFD4CE;--acc:#2B5D6B;--ok:#2F7D4F;--card:#fff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15181A;--ink:#E6E9E7;--ink2:#A7B0AB;--line:#333A3E;--acc:#7FB6C2;--ok:#7CCB98;--card:#1E2326}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,system-ui,sans-serif}}
.wrap{{max-width:560px;margin:0 auto;padding:16px 14px 120px}}
h1{{font-size:20px;margin:6px 0 2px}}.sub{{color:var(--ink2);font-size:14px;margin:0 0 14px}}
.par{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:0 0 12px}}
.num{{font-weight:700;color:var(--acc);font-size:13px;letter-spacing:.06em}}
.lado{{margin:8px 0}}.lado b{{display:inline-block;width:18px}}.lado span{{color:var(--ink2);font-size:14px}}
audio{{width:100%;height:36px;margin-top:4px}}
.btns{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px}}
button{{font:inherit;font-size:14px;padding:10px 4px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink)}}
button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 14px;display:flex;gap:8px;align-items:center}}
.bar .p{{flex:1;font-size:14px;color:var(--ink2)}}
.bar button{{background:var(--ok);color:#fff;border-color:var(--ok);padding:10px 14px;font-weight:600}}
.bar button:disabled{{opacity:.5}}
.nombre{{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);margin:0 0 12px}}
.hint{{font-size:13px;color:var(--ink2)}}
</style></head><body><div class="wrap">
<h1>{html.escape(titulo)}</h1>
<p class="sub">Lote {lote} · escuchá A y B de cada par y tocá cuál rescatarías. Al final, "Enviar" arma el mensaje.</p>
<input class="nombre" id="nombre" placeholder="Tu nombre (opcional)">
<div id="lista"></div>
<p class="hint">Los votos quedan guardados en este teléfono aunque cierres la página. Podés cambiar cualquiera antes de enviar.</p>
</div>
<div class="bar"><div class="p" id="prog"></div><button id="enviar" disabled>Enviar por WhatsApp</button><button id="copiar" style="background:var(--acc);border-color:var(--acc)">Copiar</button></div>
<script>
const LOTE={lote};const PARES={datos_json};
const KEY="zapaia_lote_"+LOTE;let votos={{}};try{{votos=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
const OPC=[["A","A"],["B","B"],["ambos","ambos"],["ninguno","ninguno"]];
const lista=document.getElementById("lista");
for(const p of PARES){{
  const d=document.createElement("div");d.className="par";d.id="par"+p.n;
  d.innerHTML=`<div class="num">PAR ${{String(p.n).padStart(3,"0")}}</div>
  <div class="lado"><b>A</b> <span>${{esc(p.a)}}</span><audio controls preload="none" src="${{p.ua}}"></audio></div>
  <div class="lado"><b>B</b> <span>${{esc(p.b)}}</span><audio controls preload="none" src="${{p.ub}}"></audio></div>
  <div class="btns">${{OPC.map(([v,l])=>`<button data-n="${{p.n}}" data-v="${{v}}">${{l}}</button>`).join("")}}</div>`;
  lista.appendChild(d);
}}
function esc(s){{return s.replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c])}}
document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>{{document.querySelectorAll("audio").forEach(o=>{{if(o!==a)o.pause()}})}}));
lista.addEventListener("click",e=>{{const b=e.target.closest("button");if(!b)return;
  votos[b.dataset.n]=b.dataset.v;try{{localStorage.setItem(KEY,JSON.stringify(votos))}}catch(x){{}}render();}});
function texto(){{const nom=document.getElementById("nombre").value.trim();
  const partes=PARES.filter(p=>votos[p.n]).map(p=>`${{p.n}} ${{votos[p.n]}}`);
  return `LOTE ${{LOTE}}${{nom?" ("+nom+")":""}}: `+partes.join(", ");}}
function render(){{
  document.querySelectorAll("#lista button").forEach(b=>b.classList.toggle("on",votos[b.dataset.n]===b.dataset.v));
  const n=PARES.filter(p=>votos[p.n]).length;
  document.getElementById("prog").textContent=`${{n}} de ${{PARES.length}} respondidos`;
  document.getElementById("enviar").disabled=n===0;
}}
document.getElementById("enviar").onclick=()=>{{location.href="https://wa.me/?text="+encodeURIComponent(texto())}};
document.getElementById("copiar").onclick=async()=>{{try{{await navigator.clipboard.writeText(texto());document.getElementById("prog").textContent="Copiado ✓"}}catch(e){{prompt("Copiá este texto:",texto())}}}};
render();
</script></body></html>
"""
