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
# 1. PARÁMETROS
#    CAMBIO: sigma desaparece.
#    Aparecen alpha_std y beta_std que representan
#    la incertidumbre sobre cada parámetro.
# ─────────────────────────────────────────────

alpha_hat = 200        # estimación puntual de α (demanda base)
alpha_std = 20         # desv est. sobre α (± cuánto puede variar)

beta_hat  = 0.008      # estimación puntual de β (sensibilidad al precio)
beta_std  = 0.001      # desv est. sobre β

costo     = 3_000      # costo unitario en CLP (fijo y conocido)

p_min = 5_000
p_max = 20_000
J     = 50             # tamaño de la grilla de precios

# ─────────────────────────────────────────────
# 2. GRILLA DE PRECIOS  P = {p1, p2, ..., p50}
# ─────────────────────────────────────────────

P = np.linspace(p_min, p_max, J)   # 50 precios igualmente espaciados

# ─────────────────────────────────────────────
# 3. QUINTILES  (5 escenarios + 5 reglas)
# CAMBIOS: tenemos 1.000 escenarios.
# Se muestran 1000 pares (α_k, β_k) con dist normal.
# ─────────────────────────────────────────────
S   = 1_000
rng = np.random.default_rng(seed=42)   # seed para resultados reproducibles

alphas = rng.normal(alpha_hat, alpha_std, S)   # α_k ~ N(α̂, σ_α)
betas  = rng.normal(beta_hat,  beta_std,  S)   # β_k ~ N(β̂, σ_β)
betas  = np.clip(betas, 1e-6, None)            # β no puede ser negativo

# ─────────────────────────────────────────────
# 4. FUNCIÓN DE DEMANDA
#    D(p, τ) = max(0,  α - β·p  +  σ·Φ⁻¹(τ))
#    → una curva distinta por cada τ
# ─────────────────────────────────────────────
D = np.maximum(0,
               alphas[:, None]           # columna de α_k  → shape (S, 1)
               - betas[:, None] * P[None, :])  # broadcast con P → shape (S, J)

# ─────────────────────────────────────────────
# 5. GRÁFICO DE CURVAS DE DEMANDA
#    Graficamos 30 escenarios de muestra (graficar 1000 sería ilegible)
#    y encima la curva de la estimación puntual (α̂, β̂).
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(30):
    ax.plot(P, D[k], linewidth=0.8, alpha=0.4, color="#60a5fa")

D_hat = np.maximum(0, alpha_hat - beta_hat * P)   # curva con valores estimados
ax.plot(P, D_hat, linewidth=2.5, color="white",
        label=r"$D(p;\hat{\alpha},\hat{\beta})$ — estimación puntual")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Demanda (unidades)")
ax.set_title("Muestra de curvas de demanda (30 de 1000 escenarios)\n"
             r"$D(p;\alpha_k,\beta_k)=\max(0,\alpha_k-\beta_k\,p)$", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("curvas_demanda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: curvas_demanda.png")

# ─────────────────────────────────────────────
# 7. PROFIT
#    Π(p, τ) = D(p, τ) · (p - costo)
# ─────────────────────────────────────────────

# Matriz de profit: Pi[s][j] = profit del escenario s al precio P[j]
Pi = D * (P[None, :] - costo)    # shape (S, J)

# ─────────────────────────────────────────────
# 8. PRECIO ÓPTIMO
#    NUEVO: ya no solo calculamos p* por escenario.
#    Calculamos el profit ESPERADO E[Π(p)] promediando sobre todos los escenarios,
#    y encontramos el precio que lo maximiza.
#
#    E[Π(P[j])] = (1/S) · Σ_k Π[k, j]
#    p_EV = argmax_j E[Π(P[j])]
# ─────────────────────────────────────────────
E_Pi     = Pi.mean(axis=0)        # promedio sobre escenarios → shape (J,)
p_EV_idx = np.argmax(E_Pi)
p_EV     = P[p_EV_idx]

print(f"\n→ Precio que maximiza el profit esperado:  p_EV = ${p_EV:,.0f} CLP")
print(f"  Profit esperado en p_EV = ${E_Pi[p_EV_idx]:,.0f}")

# ─────────────────────────────────────────────
# 9. GRÁFICO DE PROFIT CON p* MARCADO
#    Mostramos escenarios de fondo + curva del valor esperado.
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(30):
    ax.plot(P, Pi[k], linewidth=0.8, alpha=0.3, color="#86efac")

ax.plot(P, E_Pi, linewidth=2.5, color="white",
        label=r"$\mathbb{E}[\Pi(p)]$ — profit esperado")
ax.axvline(p_EV,  color="gold",  linestyle="--", linewidth=1.5,
           label=f"p_EV  = ${p_EV:,.0f}")
ax.axvline(costo, color="gray",  linestyle=":",  linewidth=1,
           label=f"costo = ${costo:,}")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Profit (CLP)")
ax.set_title("Curvas de profit por escenario\n"
             r"$\Pi(p;\alpha_k,\beta_k)=D(p;\alpha_k,\beta_k)\cdot(p-c)$", pad=14)
ax.legend(loc="upper left")
ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("curvas_profit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: curvas_profit.png")

# ─────────────────────────────────────────────
# 10. ORÁCULO
#     CAMBIO IMPORTANTE: antes el oráculo era el máximo de cada escenario
#     evaluado en su propio precio óptimo → la diagonal era trivialmente cero
#     y el ganador siempre era el escenario del medio.
#
#     Ahora separamos dos cosas:
#       (a) (α_oracle, β_oracle): el "valor verdadero" del mundo, cercano
#           pero NO igual a (α̂, β̂). Representa que nunca estimamos exacto.
#       (b) Pi_best[k]: el mejor profit posible EN el escenario k,
#           mirando toda la grilla → es el benchmark retrospectivo del regret.
# ─────────────────────────────────────────────

alpha_oracle = alpha_hat + 0.3 * alpha_std   # ligeramente distinto de α̂
beta_oracle  = beta_hat  + 0.3 * beta_std    # ligeramente distinto de β̂

print(f"\n── Oráculo (parámetros 'verdaderos') ──")
print(f"  α_oracle = {alpha_oracle:.2f}  (vs α̂ = {alpha_hat})")
print(f"  β_oracle = {beta_oracle:.5f} (vs β̂ = {beta_hat})")

# Mejor profit posible en cada escenario (máximo sobre toda la grilla de precios)
Pi_best = Pi.max(axis=1)    # shape (S,)

# ─────────────────────────────────────────────
# 11. PROFIT REALIZADO
#     Para cada precio P[j] como "regla g", su profit en cada escenario k
#     ya está calculado: es simplemente Pi[k, j].
#     Transponemos para tener shape (J, S): Pi_real[j, k].
# ─────────────────────────────────────────────

Pi_real = Pi.T    # shape (J, S):  Pi_real[j, k] = profit de la regla j en escenario k

# ─────────────────────────────────────────────
# 12. REGRET
#     R[j, k] = Pi_best[k] - Pi_real[j, k]
#       → cuánto dejaste de ganar en el escenario k al fijar el precio P[j]
#
#     R_bar[j] = promedio de R[j, k] sobre k
#       → regret esperado del precio P[j]
#
#     g* = argmin_j R_bar[j]  → precio que minimiza el regret esperado
# ─────────────────────────────────────────────

R     = Pi_best[None, :] - Pi_real   # shape (J, S)
R_bar = R.mean(axis=1)               # shape (J,)

g_star_idx = np.argmin(R_bar)
p_g_star   = P[g_star_idx]

print(f"\n→ Precio que minimiza el regret esperado:  p*(g*) = ${p_g_star:,.0f} CLP")
print(f"  Regret esperado mínimo = ${R_bar[g_star_idx]:,.0f}")

# ─────────────────────────────────────────────
# 13. GRÁFICO DE REGRET ESPERADO
#     NUEVO: visualiza la curva E[R(p)] sobre toda la grilla.
#     El mínimo es la regla óptima g*.
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(P, R_bar, linewidth=2, color="#f87171",
        label=r"$\bar{R}(p)$ — regret esperado")
ax.axvline(p_g_star, color="gold",  linestyle="--", linewidth=1.5,
           label=f"p*(g*) = ${p_g_star:,.0f}  ← regla óptima")
ax.axvline(p_EV,     color="cyan",  linestyle=":",  linewidth=1.5,
           label=f"p_EV   = ${p_EV:,.0f}  ← max profit esperado")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Regret esperado (CLP)")
ax.set_title("Regret esperado por precio\n"
             r"$\bar{R}(p)=\frac{1}{S}\sum_k\left[\Pi_k^*-\Pi(p;\alpha_k,\beta_k)\right]$",
             pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("regret_esperado.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: regret_esperado.png")