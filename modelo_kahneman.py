"""
Regret-Grid: Ejemplo Acrílico  ·  versión con AVERSIÓN A LA PÉRDIDA (Kahneman)
==============================================================================
Modelo base:   D(p) = max(0, α - β·p)
Modelo nuevo:  D(p) = max(0, α - β·p + γ·v(p; R, λ))

donde v(p; R, λ) es la función de valor de Kahneman respecto a un
precio de referencia R:

    v(p) = max(0, R - p)  -  λ · max(0, p - R)

    · si p < R  → el cliente lo percibe como GANANCIA (cuenta 1×)
    · si p > R  → lo percibe como PÉRDIDA   (cuenta λ×, con λ = 2.25)

La asimetría (λ > 1) es lo que hace que el precio que maximiza el profit
esperado (p_EV) y el que minimiza el regret esperado (p_g*) dejen de coincidir.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   # para Φ⁻¹(τ)

# ─────────────────────────────────────────────
# 1. PARÁMETROS
# ─────────────────────────────────────────────

alpha_hat = 200        # estimación puntual de α (demanda base)
alpha_std = 20         # desv est. sobre α (± cuánto puede variar)

beta_hat  = 0.008      # estimación puntual de β (sensibilidad al precio)
beta_std  = 0.001      # desv est. sobre β

costo     = 3_000      # costo unitario en CLP (fijo y conocido)

p_min = 5_000
p_max = 20_000
J     = 50             # tamaño de la grilla de precios

# ─── ★ NUEVO: parámetros de Kahneman (aversión a la pérdida) ───
lam   = 2.25           # λ: cuánto más pesa una "pérdida" que una "ganancia"
R_ref = 12_000         # precio de referencia (lo que el cliente "espera" pagar)
gamma = 0.005          # γ: escala que conecta el valor percibido con la demanda
#                        ↑ ESTE es el parámetro a calibrar ("alfa precio")

# ─────────────────────────────────────────────
# 2. GRILLA DE PRECIOS  P = {p1, p2, ..., p50}
# ─────────────────────────────────────────────

P = np.linspace(p_min, p_max, J)   # 50 precios equiespaciados

# ─────────────────────────────────────────────
# 3. ESCENARIOS  (1000 pares α_k, β_k)
# ─────────────────────────────────────────────
S   = 1_000
rng = np.random.default_rng(seed=42)   # seed para resultados reproducibles

alphas = rng.normal(alpha_hat, alpha_std, S)   # α_k ~ N(α̂, σ_α)
betas  = rng.normal(beta_hat,  beta_std,  S)   # β_k ~ N(β̂, σ_β)
betas  = np.clip(betas, 1e-6, None)            # β no puede ser negativo

# ─────────────────────────────────────────────
# ★ NUEVO: FUNCIÓN DE VALOR DE KAHNEMAN
#   Vectorizada: p puede ser un escalar o un array (broadcasting).
#   Devuelve un valor que se SUMA a la demanda:
#     · positivo cuando el precio está bajo la referencia
#     · negativo (y amplificado por λ) cuando está sobre la referencia
# ─────────────────────────────────────────────
def valor_kahneman(p, R, lam):
    ganancia = np.maximum(0, R - p)    # precio bajo R → ganancia percibida
    perdida  = np.maximum(0, p - R)    # precio sobre R → pérdida percibida
    return ganancia - lam * perdida    # la pérdida pesa lam veces más

# ─────────────────────────────────────────────
# 4. FUNCIÓN DE DEMANDA
#    ANTES:  D = max(0, α_k - β_k·p)
#    AHORA:  D = max(0, α_k - β_k·p + γ·v(p; R, λ))   ★
#    El término de Kahneman es común a todos los escenarios:
#    depende solo del precio (es un rasgo del consumidor, no del escenario).
# ─────────────────────────────────────────────
D = np.maximum(0,
               alphas[:, None]                       # α_k → shape (S, 1)
               - betas[:, None] * P[None, :]         # β_k·p → broadcast (S, J)
               + gamma * valor_kahneman(P[None, :], R_ref, lam))  # ★ término nuevo

# ─────────────────────────────────────────────
# 5. GRÁFICO DE CURVAS DE DEMANDA
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(50):
    ax.plot(P, D[k], linewidth=0.8, alpha=0.4, color="#60a5fa")

# ★ la curva puntual también incluye ahora el término de Kahneman
D_hat = np.maximum(0, alpha_hat - beta_hat * P
                   + gamma * valor_kahneman(P, R_ref, lam))
ax.plot(P, D_hat, linewidth=2.5, color="#ffde5b",
        label=r"$D(p;\hat{\alpha},\hat{\beta})$ — estimación puntual")

# ★ marcamos el precio de referencia R
ax.axvline(R_ref, color="#a78bfa", linestyle="--", linewidth=1.5,
           label=f"R = ${R_ref:,.0f}  ← precio de referencia")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Demanda (unidades)")
ax.set_title("Curvas de demanda con aversión a la pérdida (50 de 1000 escenarios)\n"
             r"$D(p)=\max(0,\alpha-\beta p+\gamma\,v(p;R,\lambda))$", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("curvas_demanda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: curvas_demanda.png")

# ─────────────────────────────────────────────
# 7. PROFIT
#    Π(p) = D(p) · (p - costo)   ← sin cambios: la asimetría ya entró por D
# ─────────────────────────────────────────────
Pi = D * (P[None, :] - costo)    # shape (S, J)

# ─────────────────────────────────────────────
# 8. PRECIO ÓPTIMO  (max profit esperado)
# ─────────────────────────────────────────────
E_Pi     = Pi.mean(axis=0)        # promedio sobre escenarios → shape (J,)
p_EV_idx = np.argmax(E_Pi)
p_EV     = P[p_EV_idx]

print(f"\n→ Precio que maximiza el profit esperado:  p_EV = ${p_EV:,.0f} CLP")
print(f"  Profit esperado en p_EV = ${E_Pi[p_EV_idx]:,.0f}")

# ─────────────────────────────────────────────
# 9. GRÁFICO DE PROFIT CON p* MARCADO
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in range(50):
    ax.plot(P, Pi[k], linewidth=0.8, alpha=0.3, color="#4a9b68")

ax.plot(P, E_Pi, linewidth=2.5, color="#427de3",
        label=r"$\mathbb{E}[\Pi(p)]$ — profit esperado")
ax.axvline(p_EV,  color="gold",  linestyle="--", linewidth=1.5,
           label=f"p_EV  = ${p_EV:,.0f}")
ax.axvline(R_ref, color="#a78bfa", linestyle="--", linewidth=1,    # ★ referencia
           label=f"R = ${R_ref:,.0f}")
ax.axvline(costo, color="gray",  linestyle=":",  linewidth=1,
           label=f"costo = ${costo:,}")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Profit (CLP)")
ax.set_title("Curvas de profit por escenario (con aversión a la pérdida)", pad=14)
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
#     (sin cambios respecto a tu versión: el oráculo y el benchmark
#      retrospectivo son independientes de la forma de la demanda)
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
# ─────────────────────────────────────────────
Pi_real = Pi.T    # shape (J, S):  Pi_real[j, k] = profit de la regla j en escenario k

# ─────────────────────────────────────────────
# 12. REGRET
# ─────────────────────────────────────────────
R     = Pi_best[None, :] - Pi_real   # shape (J, S)
R_bar = R.mean(axis=1)               # shape (J,)

g_star_idx = np.argmin(R_bar)
p_g_star   = P[g_star_idx]

print(f"\n→ Precio que minimiza el regret esperado:  p*(g*) = ${p_g_star:,.0f} CLP")
print(f"  Regret esperado mínimo = ${R_bar[g_star_idx]:,.0f}")

# ─── ★ NUEVO: comparación directa de los dos criterios ───
print(f"\n── Comparación de criterios ──")
print(f"  p_EV  (max profit esperado)  = ${p_EV:,.0f}")
print(f"  p_g*  (min regret esperado)  = ${p_g_star:,.0f}")
print(f"  Diferencia                   = ${abs(p_EV - p_g_star):,.0f}")

# ─────────────────────────────────────────────
# 13. GRÁFICO DE REGRET ESPERADO
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(P, R_bar, linewidth=2, color="#f87171",
        label=r"$\bar{R}(p)$ — regret esperado")
ax.axvline(p_g_star, color="gold",  linestyle="--", linewidth=1.5,
           label=f"p*(g*) = ${p_g_star:,.0f}  ← regla óptima")
ax.axvline(p_EV,     color="cyan",  linestyle=":",  linewidth=1.5,
           label=f"p_EV   = ${p_EV:,.0f}  ← max profit esperado")

ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Regret esperado (CLP)")
ax.set_title("Regret esperado por precio (con aversión a la pérdida)", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("regret_esperado.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: regret_esperado.png")