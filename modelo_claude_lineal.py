"""
Regret-Grid: Ejemplo Acrílico
==============================
Modelo: D(p, τ) = max(0, α - β·p + σ·Φ⁻¹(τ))

Φ⁻¹(τ) = inversa de la normal estándar (viene de scipy)
En la tesis real, este bloque lo reemplaza la red neuronal.

Precios en CLP.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   # para Φ⁻¹(τ)

# ─────────────────────────────────────────────
# 1. PARÁMETROS  →  juega cambiando estos valores
# ─────────────────────────────────────────────

alpha = 200        # demanda base, para precio = 0
beta  = 0.008      # sensibilidad al precio: por cada $1 CLP extra, se pierden β unidades
sigma = 20         # ruido: qué tan dispersos son los escenarios entre sí
costo = 3_000      # costo unitario en CLP

p_min = 5_000      # precio mínimo de la grilla (CLP)
p_max = 20_000     # precio máximo de la grilla (CLP)
J     = 50         # cuántos precios distintos tiene la grilla

# ─────────────────────────────────────────────
# 2. GRILLA DE PRECIOS  P = {p1, p2, ..., p50}
# ─────────────────────────────────────────────

P = np.linspace(p_min, p_max, J)   # 50 precios igualmente espaciados

# ─────────────────────────────────────────────
# 3. QUINTILES  (5 escenarios + 5 reglas)
# ─────────────────────────────────────────────

taus   = [0.1, 0.3, 0.5, 0.7, 0.9]
labels = ["τ=0.1 (pesimista)", "τ=0.3", "τ=0.5 (mediana)", "τ=0.7", "τ=0.9 (optimista)"]
colors = ["#f87171", "#fb923c", "#fbbf24", "#86efac", "#34d399"]

# ─────────────────────────────────────────────
# 4. FUNCIÓN DE DEMANDA
#    D(p, τ) = max(0,  α - β·p  +  σ·Φ⁻¹(τ))
#    → una curva distinta por cada τ
# ─────────────────────────────────────────────

def demanda(p, tau):
    z = norm.ppf(tau)           # Φ⁻¹(τ): z-score del cuantil
    return np.maximum(0, alpha - beta * p + sigma * z)

# Matriz de demanda: D[s][j] = demanda del escenario s al precio P[j]
taus   = [0.1, 0.3, 0.5, 0.7, 0.9]
D = np.array([demanda(P, tau) for tau in taus])   # shape (5, 50)

# ─────────────────────────────────────────────
# 5. GRÁFICO DE CURVAS DE DEMANDA
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for s in range(5):
    ax.plot(P, D[s], linewidth=2, label=labels[s])

ax.set_xlabel("Precio (CLP)", fontsize=11)
ax.set_ylabel("Demanda (unidades)", fontsize=11)
ax.set_title("Curvas de demanda por quintil\n"
             r"$D(p,\tau) = \max(0,\ \alpha - \beta p + \sigma \cdot \Phi^{-1}(\tau))$",
             fontsize=13, pad=14)

ax.legend(loc="upper right", fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(linewidth=0.5, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("curvas_demanda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado en curvas_demanda.png")

# ─────────────────────────────────────────────
# 6. IMPRIME TABLA DE DEMANDA (opcional)
#    para ver los números crudos
# ─────────────────────────────────────────────

print("\n── Demanda en precios seleccionados ──")
check_prices = [5_000, 8_000, 10_000, 13_000, 16_000, 20_000]
header = f"{'Precio':>10}" + "".join(f"  {t:>8}" for t in taus)
print(header)
print("-" * (10 + 10 * 5))
for p in check_prices:
    row = f"${p:>9,.0f}"
    for tau in taus:
        d = max(0, alpha - beta * p + sigma * norm.ppf(tau))
        row += f"  {d:>8.1f}"
    print(row)


# ─────────────────────────────────────────────
# 7. PROFIT
#    Π(p, τ) = D(p, τ) · (p - costo)
# ─────────────────────────────────────────────

# Matriz de profit: Pi[s][j] = profit del escenario s al precio P[j]
Pi = np.array([D[s] * (P - costo) for s in range(5)])   # shape (5, 50)

# ─────────────────────────────────────────────
# 8. PRECIO ÓPTIMO ANALÍTICO
#    p* = (α + β·c + σ·z) / (2β)
# ─────────────────────────────────────────────

p_star = []
for tau in taus:
    z = norm.ppf(tau)
    p_opt = (alpha + beta * costo + sigma * z) / (2 * beta)
    p_star.append(p_opt)

# profit en cada p* (enchufamos p* en la fórmula de profit)
pi_star = []
for s in range(5):
    z     = norm.ppf(taus[s])
    d_opt = max(0, alpha - beta * p_star[s] + sigma * z)
    pi_star.append(d_opt * (p_star[s] - costo))

# ─────────────────────────────────────────────
# 9. GRÁFICO DE PROFIT CON p* MARCADO
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for s in range(5):
    ax.plot(P, Pi[s], linewidth=2, label=labels[s])

# Marcar el p* de cada escenario con un punto
for s in range(5):
    ax.plot(p_star[s], pi_star[s], marker="o", markersize=8, color="black")
    ax.annotate(f"p*={p_star[s]:,.0f}",
                xy=(p_star[s], pi_star[s]),
                xytext=(0, 10), textcoords="offset points",
                fontsize=8, ha="center")

ax.axvline(x=costo, color="gray", linestyle=":", linewidth=1, label=f"costo = ${costo:,}")

ax.set_xlabel("Precio (CLP)", fontsize=11)
ax.set_ylabel("Profit (CLP)", fontsize=11)
ax.set_title("Curvas de profit por quintil\n"
             r"$\Pi(p,\tau) = D(p,\tau) \cdot (p - c)$",
             fontsize=13, pad=14)

ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(linewidth=0.5, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("curvas_profit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado en curvas_profit.png")

# ─────────────────────────────────────────────
# 10. IMPRIME TABLA DE p* Y PROFIT ÓPTIMO
# ─────────────────────────────────────────────

print("\n── Precio óptimo por regla g ──")
print(f"{'Regla g':>10}  {'z':>6}  {'p*':>12}  {'Profit en p*':>15}")
print("-" * 50)
for s in range(5):
    z = norm.ppf(taus[s])
    print(f"  g={taus[s]:>3}    {z:>6.2f}  ${p_star[s]:>10,.0f}  ${pi_star[s]:>13,.0f}")

# ─────────────────────────────────────────────
# 11. PROFIT REALIZADO
#     Para cada regla g, evaluamos su p*(g)
#     en CADA escenario s
#     Pi_real[g][s] = D[s](p*(g)) * (p*(g) - costo)
# ─────────────────────────────────────────────

# Primero necesitamos el índice de p*(g) en la grilla P
# (buscamos el precio más cercano en P a cada p* analítico)
p_star_idx = []
for s in range(5):
    idx = np.argmin(np.abs(P - p_star[s]))
    p_star_idx.append(idx)

# Ahora calculamos el profit realizado
Pi_real = np.zeros((5, 5))   # Pi_real[g][s]

for g in range(5):            # para cada regla
    for s in range(5):        # en cada escenario
        d_realizada = D[s][p_star_idx[g]]          # demanda del escenario s al precio p*(g)
        Pi_real[g][s] = d_realizada * (P[p_star_idx[g]] - costo)

# ─────────────────────────────────────────────
# 12. ORÁCULO
#     Pi_best[s] = max sobre toda la grilla P
#                  del profit en el escenario s
# ─────────────────────────────────────────────

Pi_best = np.zeros(5)         # un valor por escenario

for s in range(5):
    Pi_best[s] = np.max(Pi[s])    # Pi[s] ya lo calculamos antes

# ─────────────────────────────────────────────
# 13. REGRET
#     R[g][s] = Pi_best[s] - Pi_real[g][s]
# ─────────────────────────────────────────────

R = np.zeros((5, 5))

for g in range(5):
    for s in range(5):
        R[g][s] = Pi_best[s] - Pi_real[g][s]

# Regret promedio por regla
R_bar = R.mean(axis=1)        # axis=1 promedia sobre los escenarios

# Regla ganadora
g_star = np.argmin(R_bar)

# ─────────────────────────────────────────────
# 14. TABLA DE REGRET
# ─────────────────────────────────────────────

print("\n── Tabla de Regret R[g][s] ──")
header = f"{'Regla g':>10}" + "".join(f"  s={s+1}(τ={taus[s]})  " for s in range(5)) + "  R̄_g"
print(header)
print("-" * 80)

for g in range(5):
    marca = " ← g*" if g == g_star else ""
    fila = f"  g={taus[g]:>3} "
    for s in range(5):
        fila += f"  ${R[g][s]:>10,.0f}  "
    fila += f"  ${R_bar[g]:>10,.0f}{marca}"
    print(fila)

print(f"\n→ Regla óptima: g* = {taus[g_star]}")
print(f"→ Precio recomendado: p*(g*) = ${P[p_star_idx[g_star]]:,.0f} CLP")