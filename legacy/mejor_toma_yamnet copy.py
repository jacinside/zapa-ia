import re, pandas as pd, math

with open('/Users/javiercordero/workspace/jac/zapa-ia/log.txt', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f]

rows, current = [], None
for line in lines:
    m = re.match(r"Procesando:\s+(.*\.mp3)$", line, re.IGNORECASE)
    if m:
        current = m.group(1)
        continue
    m = re.match(r"score:\s+([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
    if m and current:
        rows.append({'file': current, 'score': float(m.group(1))})
        current = None
        continue
    if line.startswith("❌ Error procesando") and current:
        rows.append({'file': current, 'score': math.nan})
        current = None

df = pd.DataFrame(rows, columns=['file','score'])
df.to_csv('scores.csv', index=False,decimal=',')
print("OK → scores.csv")
