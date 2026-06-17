import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. PARÁMETROS
#    CAMBIO vs código base:
#    - sale β (sensibilidad lineal simétrica)
#    - entra α_p (sensibilidad al precio de Kahneman, el único que se estima)
#    - entra λ = 2.25 (aversión a la pérdida, calibrado — NO se estima)
#    - entra r_ref (precio de referencia del cliente)
# ─────────────────────────────────────────────
alpha_hat   = 200      # α̂: demanda base estimada
alpha_std   = 20       # σ_α: incertidumbre sobre el nivel

alpha_p_hat = 0.008    # α̂_p: sensibilidad marginal al precio (se estima)
alpha_p_std = 0.003    # σ_αp: incertidumbre sobre la pendiente (clave p/ minimax)

lam         = 2.25     # λ: coeficiente de aversión a la pérdida (Kahneman-Tversky)
r_ref       = 12_000   # r: precio de referencia del cliente

costo       = 3_000    # costo unitario CLP (fijo y conocido)

p_min, p_max, J = 5_000, 20_000, 50

# ─────────────────────────────────────────────
# 2. GRILLA DE PRECIOS
# ─────────────────────────────────────────────
P = np.linspace(p_min, p_max, J)

# ─────────────────────────────────────────────
# 3. ESCENARIOS: 1000 pares (α_k, α_p,k)
#    Incertidumbre en NIVEL y en PENDIENTE (como en v2).
# ─────────────────────────────────────────────
S   = 1_000
rng = np.random.default_rng(seed=42)

alphas   = rng.normal(alpha_hat,   alpha_std,   S)
alpha_ps = rng.normal(alpha_p_hat, alpha_p_std, S)
alpha_ps = np.clip(alpha_ps, 1e-6, None)          # α_p no puede ser negativo

# ─────────────────────────────────────────────
# 4. UTILIDAD TRANSACCIONAL DE KAHNEMAN
#    v(p) = [r − p]⁺ − λ·[p − r]⁺
#    Bajo r_ref: ganancia percibida (pendiente +1)
#    Sobre r_ref: pérdida percibida amplificada por λ (pendiente −λ)
# ─────────────────────────────────────────────
def valor_referencia(p, r, lam):
    return np.maximum(0, r - p) - lam * np.maximum(0, p - r)

V = valor_referencia(P[None, :], r_ref, lam)       # shape (1, J)

# ─────────────────────────────────────────────
# 5. DEMANDA
#    ANTES:  D = max(0, α_k − β_k·p)
#    AHORA:  D = max(0, α_k + α_p,k · v(p))
# ─────────────────────────────────────────────
D = np.maximum(0, alphas[:, None] + alpha_ps[:, None] * V)   # shape (S, J)

fig, ax = plt.subplots(figsize=(10, 6))
for k in range(50):
    ax.plot(P, D[k], linewidth=0.8, alpha=0.4, color="#60a5fa")
D_hat = np.maximum(0, alpha_hat + alpha_p_hat * valor_referencia(P, r_ref, lam))
ax.plot(P, D_hat, linewidth=2.5, color="#ffde5b",
        label=r"$D(p;\hat{\alpha},\hat{\alpha}_p)$ — estimación puntual")
ax.axvline(r_ref, color="#a78bfa", linestyle="--", linewidth=1.5,
           label=f"r = ${r_ref:,.0f} ← precio de referencia")
ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Demanda (unidades)")
ax.set_title("Demanda con aversión a la pérdida (50 de 1000 escenarios)\n"
             r"$D(p)=\max(0,\ \alpha+\alpha_p([r-p]^+-\lambda[p-r]^+))$", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout(); plt.savefig("curvas_demanda.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 6. PROFIT   Π(p) = D(p)·(p − c)
# ─────────────────────────────────────────────
Pi = D * (P[None, :] - costo)                      # shape (S, J)

# ─────────────────────────────────────────────
# 7. PRECIO QUE MAXIMIZA EL PROFIT ESPERADO
# ─────────────────────────────────────────────
E_Pi     = Pi.mean(axis=0)
p_EV_idx = np.argmax(E_Pi)
p_EV     = P[p_EV_idx]
print(f"→ p_EV (max profit esperado)        = ${p_EV:,.0f}")

# ─────────────────────────────────────────────
# 8. ORÁCULO Y REGRET
#    Pi_best[k] = mejor profit alcanzable en el escenario k (hindsight)
#    R[j,k]     = Pi_best[k] − Pi[k,j]
# ─────────────────────────────────────────────
Pi_best = Pi.max(axis=1)                           # shape (S,)
R       = Pi_best[None, :] - Pi.T                  # shape (J, S)

# Criterio 1: regret ESPERADO → coincide con p_EV (verificado en v2)
R_bar      = R.mean(axis=1)
g_star_idx = np.argmin(R_bar)
p_g_star   = P[g_star_idx]
print(f"→ p_g*  (min regret esperado)       = ${p_g_star:,.0f}")

# Criterio 2: regret MINIMAX → aquí sí diverge (requiere σ_αp > 0)
R_worst     = R.max(axis=1)
g_worst_idx = np.argmin(R_worst)
p_g_worst   = P[g_worst_idx]
print(f"→ p_g_worst (minimax regret)        = ${p_g_worst:,.0f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(P, R_bar,   linewidth=2, color="#f87171", label=r"$\bar{R}(p)$ — regret esperado")
ax.plot(P, R_worst, linewidth=2, color="#fb923c", label=r"$R^{max}(p)$ — regret peor caso")
ax.axvline(p_g_star,  color="gold", linestyle="--", linewidth=1.5,
           label=f"p_g* = ${p_g_star:,.0f}")
ax.axvline(p_g_worst, color="#34d399", linestyle="--", linewidth=1.5,
           label=f"p_g_worst = ${p_g_worst:,.0f}")
ax.set_xlabel("Precio (CLP)"); ax.set_ylabel("Regret (CLP)")
ax.set_title("Regret esperado vs minimax", pad=14)
ax.legend(); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout(); plt.savefig("regret_comparacion.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 9. HISTOGRAMAS: CORTE VERTICAL EN EL PRECIO ÓPTIMO
#    Idea: fijamos p = p_g* y p = p_g_worst, y miramos POR DÓNDE
#    cruzan las 1000 curvas esa línea vertical.
#      D[:, idx]  → demanda de cada escenario en ese precio  (1000 valores)
#      Pi[:, idx] → profit de cada escenario en ese precio   (1000 valores)
#    OJO: Π(p_fijo) = D(p_fijo)·(p_fijo − c) → misma forma, distinta escala.
# ─────────────────────────────────────────────
D_corte_star,  D_corte_worst  = D[:, g_star_idx],  D[:, g_worst_idx]
Pi_corte_star, Pi_corte_worst = Pi[:, g_star_idx], Pi[:, g_worst_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── (a) demanda en el corte ──
ax = axes[0]
ax.hist(D_corte_star,  bins=40, alpha=0.55, color="#60a5fa",
        label=f"D(p_g* = ${p_g_star:,.0f})")
ax.hist(D_corte_worst, bins=40, alpha=0.55, color="#34d399",
        label=f"D(p_g_worst = ${p_g_worst:,.0f})")
ax.axvline(D_corte_star.mean(),  color="#1d4ed8", linestyle="--", linewidth=1.5,
           label=f"media = {D_corte_star.mean():.1f}")
ax.axvline(D_corte_worst.mean(), color="#047857", linestyle="--", linewidth=1.5,
           label=f"media = {D_corte_worst.mean():.1f}")
ax.set_xlabel("Demanda (unidades)"); ax.set_ylabel("N° de escenarios")
ax.set_title("Distribución de la demanda en el corte\n"
             "(por dónde cruzan las 1000 curvas la vertical del precio óptimo)")
ax.legend(fontsize=9); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)

# ── (b) profit en el corte ──
ax = axes[1]
ax.hist(Pi_corte_star,  bins=40, alpha=0.55, color="#f87171",
        label=f"Π(p_g* = ${p_g_star:,.0f})")
ax.hist(Pi_corte_worst, bins=40, alpha=0.55, color="#fbbf24",
        label=f"Π(p_g_worst = ${p_g_worst:,.0f})")
ax.axvline(Pi_corte_star.mean(),  color="#b91c1c", linestyle="--", linewidth=1.5,
           label=f"media = ${Pi_corte_star.mean():,.0f}")
ax.axvline(Pi_corte_worst.mean(), color="#b45309", linestyle="--", linewidth=1.5,
           label=f"media = ${Pi_corte_worst.mean():,.0f}")
ax.set_xlabel("Profit (CLP)"); ax.set_ylabel("N° de escenarios")
ax.set_title("Distribución del profit en el corte\n"
             "(misma forma que demanda: Π = D·(p−c) con p fijo)")
ax.legend(fontsize=9); ax.grid(linewidth=0.4, linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("histogramas_corte.png", dpi=150, bbox_inches="tight")
plt.close()

# Estadísticos del corte (para el informe)
def resumen(nombre, x, en_clp=False):
    f = (lambda v: f"${v:,.0f}") if en_clp else (lambda v: f"{v:,.1f}")
    print(f"  {nombre:28s} media={f(x.mean())}  σ={f(x.std())}  "
          f"P5={f(np.percentile(x,5))}  P95={f(np.percentile(x,95))}")

print("\n── Corte en p_g* y p_g_worst ──")
resumen("D(p_g*)",        D_corte_star)
resumen("D(p_g_worst)",   D_corte_worst)
resumen("Π(p_g*)",        Pi_corte_star,  en_clp=True)
resumen("Π(p_g_worst)",   Pi_corte_worst, en_clp=True)