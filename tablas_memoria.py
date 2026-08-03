# ══════════════════════════════════════════════════════════════════════════════
#  MATERIAL COMPLEMENTARIO DE LA MEMORIA
#  Tabla 5.1 · Tabla 5.3 · Ilustración 4.1
#  ────────────────────────────────────────────────────────────────────────────
#  Dos modos de uso (el bloque detecta solo en cuál está):
#   (a) PEGADO AL FINAL de modelo_final.py → reutiliza las variables ya
#       calculadas. No re-muestrea nada. Pegar ANTES del plt.show() final del
#       modelo; si va después, ese show() bloquea la ejecución de este bloque.
#   (b) Como archivo aparte (tablas_memoria.py) junto a modelo_final.py.
#  En ambos casos los resultados provienen de los MISMOS 1.000 escenarios
#  (semilla 42) del resto de la memoria. modelo_final.py no se modifica.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt

_STANDALONE = False
try:                                   # modo (a): las variables ya existen
    profit
except NameError:                      # modo (b): se importan del modelo congelado
    _STANDALONE = True
    from modelo_final import (profit, precios, alpha, alpha_p, valor_kahneman,
                              lam, r_ref, costo, P_MIN, P_MAX, clp)
    plt.close("all")                   # descarta las figuras que crea el import

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA 5.1 · Dispersión del profit entre escenarios según el precio
#  σ del profit entre los 1.000 escenarios, y magnitud del valor percibido.
#  Sustenta el argumento de que el abanico se abre más rápido sobre r.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Tabla 5.1 · Dispersión del profit entre escenarios ──")
print(f"{'Precio':>14} | {'σ del profit':>14} | {'|v(p)|':>10}")
print("-" * 46)
for p_obj in [6_000, 9_000, 12_000, 15_000, 18_000]:
    j = int(np.argmin(np.abs(precios - p_obj)))          # punto de grilla más cercano
    sd = profit[:, j].std()                              # σ entre escenarios en ese precio
    v_abs = abs(max(0, r_ref - precios[j]) - lam * max(0, precios[j] - r_ref))
    etiqueta = f"${p_obj:,.0f}" + (" (= r)" if p_obj == r_ref else "")
    print(f"{etiqueta:>14} | ${sd:>13,.0f} | {v_abs:>10,.0f}")
print("Nota: σ calculada sobre los 1.000 escenarios en el punto de grilla")
print("más cercano a cada precio; |v(p)| según la ecuación (4.1).")

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA 5.3 · Sensibilidad de los criterios al techo de la grilla
#  Se reconstruye la grilla para cada techo manteniendo el paso en ~$30, porque
#  con N fijo el paso cambiaría y contaminaría la comparación. Solo lectura.
# ══════════════════════════════════════════════════════════════════════════════
PASO_53 = 30.06                                          # paso oficial: 15.000/499

def _criterios(p_max):
    n = int(round((p_max - P_MIN) / PASO_53)) + 1
    p = np.linspace(P_MIN, p_max, n)
    v = np.maximum(0, r_ref - p) - lam * np.maximum(0, p - r_ref)
    d = np.maximum(0, alpha[:, None] + alpha_p[:, None] * v[None, :])
    pi = d * (p - costo)[None, :]
    reg = pi.max(axis=1, keepdims=True) - pi             # oráculo por escenario
    return p[int(np.argmin(reg.mean(axis=0)))], p[int(np.argmin(reg.max(axis=0)))]

print("\n── Tabla 5.3 · Sensibilidad al techo de la grilla ──")
print(f"{'P_MAX':>10} | {'criterio esperado':>18} | {'criterio peor caso':>18}")
print("-" * 54)
for p_max in [18_000, 20_000, 22_000, 25_000, 30_000]:
    pe, pm = _criterios(p_max)
    print(f"${p_max:>9,} | ${pe:>17,.0f} | ${pm:>17,.0f}")
print("Nota: paso de grilla mantenido en ~$30 (N crece con el rango).")
print("Los precios se reportan con la precisión que el paso permite [D2].")

# ══════════════════════════════════════════════════════════════════════════════
#  ILUSTRACIÓN 4.1 · Utilidad transaccional v(p) = [r − p]⁺ − λ·[p − r]⁺
#  Se incluye el caso λ=1 como referencia visual: ambas curvas coinciden bajo r
#  y solo se separan sobre r, lo que hace visible que toda la asimetría vive en
#  la región de pérdida percibida.
# ══════════════════════════════════════════════════════════════════════════════
_p = np.linspace(P_MIN, P_MAX, 1000)

fig_i41, ax = plt.subplots(figsize=(10, 6))

ax.plot(_p, valor_kahneman(_p, lam=1.0), lw=1.6, ls=":", color="#94a3b8",
        label="caso simétrico  λ = 1   (v(p) = r − p, sin quiebre)")
ax.plot(_p, valor_kahneman(_p, lam=lam), lw=2.8, color="#6366f1",
        label=f"aversión a la pérdida  λ = {lam}")
ax.axvline(r_ref, color="#a78bfa", ls="--", lw=1.5,
           label=f"precio de referencia  r = ${r_ref:,.0f}")
ax.axhline(0, color="black", lw=0.8, alpha=0.5)

ax.set(title="Utilidad transaccional  v(p) = [r − p]⁺ − λ·[p − r]⁺",
       xlabel="Precio (CLP)", ylabel="Valor percibido  v(p)")
ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax.grid(ls="--", lw=0.4, alpha=0.4)
ax.xaxis.set_major_formatter(clp)
fig_i41.tight_layout()

fig_i41.savefig("ilustracion_4_1_utilidad_transaccional.png", dpi=200)
fig_i41.savefig("ilustracion_4_1_utilidad_transaccional.pdf")   # vectorial para el documento
print(f"\n── Ilustración 4.1 · generada con λ={lam}, r=${r_ref:,.0f}, "
      f"rango [${P_MIN:,} – ${P_MAX:,}] ──")
print("Archivos: ilustracion_4_1_utilidad_transaccional.png / .pdf")

if _STANDALONE:      # en modo (a) el show() lo hace modelo_final.py
    plt.show()