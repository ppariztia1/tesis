"""
Regret-Grid: Ejemplo Acrílico  ·  con AVERSIÓN A LA PÉRDIDA (Kahneman)
=====================================================================
Demanda nueva:  D(p) = max(0, α + α_p·( [r-p]⁺ - λ·[p-r]⁺ ))
con la utilidad transaccional de la ecuación (1):

    u(r, p) = α_p·[r - p]⁺ - λ·α_p·[p - r]⁺
    · β⁺ = α_p     (pendiente cuando p < r, zona de ganancia)
    · β⁻ = λ·α_p   (pendiente cuando p > r, zona de pérdida)

INCERTIDUMBRE: vive en α (nivel) Y en α_p (pendiente/sensibilidad).
   → La incertidumbre en α_p es la que, junto al criterio de PEOR CASO,
     hace divergir el precio del óptimo de valor esperado.
   → Si pones alpha_p_std = 0, recuperas la versión donde todo coincide.

DOS criterios de decisión:
    · regret ESPERADO  → p_g*       (coincide con p_EV por construcción)
    · regret PEOR CASO → p_g_worst  (diverge: acá se ve el efecto)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ─────────────────────────────────────────────
# 1. PARÁMETROS
# ─────────────────────────────────────────────
alpha_hat = 200        # estimación puntual de α (demanda base)
alpha_std = 20         # desv est. sobre α

costo     = 3_000      # costo unitario en CLP

p_min = 5_000
p_max = 20_000
J     = 50

# ─── Parámetros de Kahneman ───
alpha_p_hat = 0.008    # α_p: sensibilidad marginal al precio (β⁺). El único que se estima.
alpha_p_std = 0.003    # incertidumbre sobre α_p  (pon 0 para que todo coincida)
lam         = 2.25     # λ: coeficiente de aversión a la pérdida (calibrado)
r_ref       = 12_000   # r: precio de referencia del cliente

# ─────────────────────────────────────────────
# 2. GRILLA DE PRECIOS
# ─────────────────────────────────────────────
P = np.linspace(p_min, p_max, J)

# ─────────────────────────────────────────────
# 3. ESCENARIOS (1000 pares α_k, α_p_k)
# ─────────────────────────────────────────────
S   = 1_000
rng = np.random.default_rng(seed=42)

alphas   = rng.normal(alpha_hat,   alpha_std,   S)            # α_k   ~ N(α̂, σ_α)
alpha_ps = rng.normal(alpha_p_hat, alpha_p_std, S)            # α_p,k ~ N(α̂_p, σ_αp)
alpha_ps = np.clip(alpha_ps, 1e-6, None)                      # α_p no puede ser negativo

# ─────────────────────────────────────────────
#    FORMA DE REFERENCIA (la parte sin α_p)
#    v(p) = [r - p]⁺ - λ·[p - r]⁺
# ─────────────────────────────────────────────
def valor_referencia(p, r, lam):
    return np.maximum(0, r - p) - lam * np.maximum(0, p - r)

V = valor_referencia(P[None, :], r_ref, lam)                  # shape (1, J)

# ─────────────────────────────────────────────
# 4. FUNCIÓN DE DEMANDA
#    D = max(0, α_k + α_p,k · v(p))    ← sale el -β·p, entra Kahneman
# ─────────────────────────────────────────────
D = np.maximum(0, alphas[:, None] + alpha_ps[:, None] * V)    # shape (S, J)

# ─────────────────────────────────────────────
# 5. GRÁFICO DE CURVAS DE DEMANDA
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(50):
    ax.plot(P, D[k], linewidth=0.8, alpha=0.4, color="#60a5fa")

D_hat = np.maximum(0, alpha_hat + alpha_p_hat * valor_referencia(P, r_ref, lam))
ax.plot(P, D_hat, linewidth=2.5, color="#ffde5b",
        label=r"$D(p;\hat{\alpha},\hat{\alpha}_p)$ — estimación puntual")
ax.axvline(r_ref, color="#a78bfa", linestyle="--", linewidth=1.5,
           label=f"r = ${r_ref:,.0f}  ← precio de referencia")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Demanda (unidades)")
ax.set_title("Curvas de demanda con aversión a la pérdida (50 de 1000 escenarios)\n"
             r"$D(p)=\max(0,\ \alpha + \alpha_p([r-p]^+ - \lambda[p-r]^+))$", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("curvas_demanda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: curvas_demanda.png")

# ─────────────────────────────────────────────
# 7. PROFIT   Π(p) = D(p) · (p - costo)
# ─────────────────────────────────────────────
Pi = D * (P[None, :] - costo)    # shape (S, J)

# ─────────────────────────────────────────────
# 8. PRECIO QUE MAXIMIZA EL PROFIT ESPERADO
# ─────────────────────────────────────────────
E_Pi     = Pi.mean(axis=0)
p_EV_idx = np.argmax(E_Pi)
p_EV     = P[p_EV_idx]

print(f"\n→ Precio que maximiza el profit esperado:  p_EV = ${p_EV:,.0f} CLP")
print(f"  Profit esperado en p_EV = ${E_Pi[p_EV_idx]:,.0f}")

# ─────────────────────────────────────────────
# 9. GRÁFICO DE PROFIT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(50):
    ax.plot(P, Pi[k], linewidth=0.8, alpha=0.3, color="#4a9b68")

ax.plot(P, E_Pi, linewidth=2.5, color="#427de3",
        label=r"$\mathbb{E}[\Pi(p)]$ — profit esperado")
ax.axvline(p_EV,  color="gold",    linestyle="--", linewidth=1.5, label=f"p_EV = ${p_EV:,.0f}")
ax.axvline(r_ref, color="#a78bfa", linestyle="--", linewidth=1,   label=f"r = ${r_ref:,.0f}")
ax.axvline(costo, color="gray",    linestyle=":",  linewidth=1,   label=f"costo = ${costo:,}")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Profit (CLP)")
ax.set_title("Curvas de profit por escenario (con aversión a la pérdida)", pad=14)
ax.legend(loc="upper left"); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("curvas_profit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: curvas_profit.png")

# ─────────────────────────────────────────────
# 10. ORÁCULO / BENCHMARK RETROSPECTIVO
#     Pi_best[k] = mejor profit posible EN el escenario k (toda la grilla)
# ─────────────────────────────────────────────
Pi_best = Pi.max(axis=1)    # shape (S,)

# ─────────────────────────────────────────────
# 11. PROFIT REALIZADO  →  Pi_real[j, k]
# ─────────────────────────────────────────────
Pi_real = Pi.T    # shape (J, S)

# ─────────────────────────────────────────────
# 12. REGRET  —  DOS CRITERIOS
#     R[j, k] = Pi_best[k] - Pi_real[j, k]
#     (a) ESPERADO : R_bar[j] = promedio sobre k → p_g*  (= p_EV por construcción)
#     (b) PEOR CASO: R_max[j] = máximo sobre k   → p_g_worst (diverge)
# ─────────────────────────────────────────────
R = Pi_best[None, :] - Pi_real        # shape (J, S)

R_bar = R.mean(axis=1)
p_g_star = P[np.argmin(R_bar)]

R_max = R.max(axis=1)
p_g_worst = P[np.argmin(R_max)]

print(f"\n→ Regret ESPERADO  → p_g*      = ${p_g_star:,.0f} CLP  (mín = ${R_bar[np.argmin(R_bar)]:,.0f})")
print(f"→ Regret PEOR CASO → p_g_worst = ${p_g_worst:,.0f} CLP  (mín-máx = ${R_max[np.argmin(R_max)]:,.0f})")

print(f"\n── Comparación de criterios ──")
print(f"  p_EV       (max profit esperado)   = ${p_EV:,.0f}")
print(f"  p_g*       (min regret esperado)   = ${p_g_star:,.0f}   diff vs p_EV = ${abs(p_EV-p_g_star):,.0f}")
print(f"  p_g_worst  (min regret peor caso)  = ${p_g_worst:,.0f}   diff vs p_EV = ${abs(p_EV-p_g_worst):,.0f}")

# ─────────────────────────────────────────────
# 13. GRÁFICO DE REGRET (esperado vs peor caso)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(P, R_bar, linewidth=2, color="#f87171", label=r"$\bar{R}(p)$ — regret esperado")
ax.plot(P, R_max, linewidth=2, color="#b45309", linestyle="-.",
        label=r"$\max_k R(p)$ — regret peor caso")
ax.axvline(p_g_star,  color="gold",    linestyle="--", linewidth=1.5, label=f"p_g* = ${p_g_star:,.0f}")
ax.axvline(p_g_worst, color="#b45309", linestyle="--", linewidth=1.5, label=f"p_g_worst = ${p_g_worst:,.0f}")
ax.axvline(p_EV,      color="cyan",    linestyle=":",  linewidth=1.5, label=f"p_EV = ${p_EV:,.0f}")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Regret (CLP)")
ax.set_title("Regret por precio: esperado vs peor caso", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("regret_esperado.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: regret_esperado.png")