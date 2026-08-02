# ══════════════════════════════════════════════════════════════════════════════
#  TABLAS 5.1 y 5.3 DE LA MEMORIA
#  ────────────────────────────────────────────────────────────────────────────
#  Dos modos de uso (el bloque detecta solo en cuál está):
#   (a) PEGADO AL FINAL de modelo_final.py  → reutiliza las variables ya
#       calculadas (profit, precios, alpha, alpha_p, ...). No re-muestrea nada.
#   (b) Como archivo aparte (tablas_memoria.py) junto a modelo_final.py
#       → importa esas mismas variables del modelo congelado.
#  En ambos casos los resultados provienen de los MISMOS 1.000 escenarios
#  (semilla 42) del resto de la memoria.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np

try:                                   # modo (a): pegado al final del modelo
    profit                             # ¿existen ya las variables?
except NameError:                      # modo (b): archivo aparte
    import matplotlib.pyplot as plt
    from modelo_final import (profit, precios, alpha, alpha_p,
                              lam, r_ref, costo, P_MIN)
    plt.close("all")                   # cierra las figuras que crea el import

# ── TABLA 5.1 · Dispersión del profit entre escenarios según el precio ───────
#    Para cada precio de referencia de la tabla: desviación estándar del profit
#    entre los 1.000 escenarios, y magnitud del valor percibido |v(p)|.
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

# ── TABLA 5.3 · Sensibilidad de los criterios al límite superior de la grilla ─
#    Para cada P_MAX se reconstruye la grilla manteniendo el paso en ~$30
#    (N crece con el rango), y se recalculan demanda, profit, oráculo y regret
#    sobre los mismos escenarios. modelo_final.py no se modifica: esto solo lee.
PASO_53 = 30.06                                          # paso oficial: 15.000/499

def _criterios(p_max):
    n = int(round((p_max - P_MIN) / PASO_53)) + 1
    p = np.linspace(P_MIN, p_max, n)
    v = np.maximum(0, r_ref - p) - lam * np.maximum(0, p - r_ref)
    d = np.maximum(0, alpha[:, None] + alpha_p[:, None] * v[None, :])
    pi = d * (p - costo)[None, :]
    reg = pi.max(axis=1, keepdims=True) - pi             # oráculo por escenario
    return p[int(np.argmin(reg.mean(axis=0)))], p[int(np.argmin(reg.max(axis=0)))]

print("\n── Tabla 5.3 · Sensibilidad al límite superior de la grilla ──")
print(f"{'P_MAX':>10} | {'criterio esperado':>18} | {'criterio peor caso':>18}")
print("-" * 54)
for p_max in [18_000, 20_000, 22_000, 25_000, 30_000]:
    pe, pm = _criterios(p_max)
    print(f"${p_max:>9,} | ${pe:>17,.0f} | ${pm:>17,.0f}")
print("Nota: paso de grilla mantenido en ~$30 (N crece con el rango).")
print("Los precios se reportan con la precisión que el paso permite [D2].")