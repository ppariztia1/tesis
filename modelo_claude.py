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

alpha = 200        # demanda base (unidades) cuando el precio fuera 0
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
D = np.array([demanda(P, tau) for tau in taus])   # shape (5, 50)

# ─────────────────────────────────────────────
# 5. GRÁFICO DE CURVAS DE DEMANDA
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor("#0e1a2b")
ax.set_facecolor("#080f1a")

for s in range(5):
    ax.plot(P, D[s], color=colors[s], linewidth=2, label=labels[s])

# Anotación del z-score en el extremo derecho
for s in range(5):
    z = norm.ppf(taus[s])
    ax.annotate(f"z = {z:.2f}", xy=(P[-1], D[s][-1]),
                xytext=(5, 0), textcoords="offset points",
                color=colors[s], fontsize=9, va="center")

ax.set_xlabel("Precio (CLP)", color="#94a3b8", fontsize=11)
ax.set_ylabel("Demanda (unidades)", color="#94a3b8", fontsize=11)
ax.set_title("Curvas de demanda por quintil\n"
             r"$D(p,\tau) = \max(0,\ \alpha - \beta p + \sigma \cdot \Phi^{-1}(\tau))$",
             color="#fbbf24", fontsize=13, pad=14)

ax.legend(loc="upper right", framealpha=0.2, labelcolor="white", fontsize=9)
ax.tick_params(colors="#475569")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(color="#1e293b", linewidth=0.8)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e293b")

plt.tight_layout()
plt.savefig("/home/claude/curvas_demanda.png", dpi=150, bbox_inches="tight")
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