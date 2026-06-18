"""
═══════════════════════════════════════════════════════════════════════════
 MODELO REGRET-GRID CON AVERSIÓN A LA PÉRDIDA (Kahneman-Tversky)
 Trabajo de Título · Pricing óptimo bajo demanda incierta
───────────────────────────────────────────────────────────────────────────
 FLUJO DEL MODELO (para quien lo lee por primera vez):

   1. Definimos parámetros y una grilla de precios candidatos.
   2. Generamos 1.000 ESCENARIOS de demanda, muestreando dos fuentes de
      incertidumbre: el nivel de demanda (α) y la sensibilidad al precio (α_p).
   3. La demanda reacciona al precio vía la utilidad transaccional de Kahneman:
      el cliente compara el precio con un precio de referencia r, y penaliza
      MÁS las "pérdidas" (p > r) que lo que premia las "ganancias" (p < r).
      Esa asimetría es el coeficiente λ = 2.25.
   4. Para cada escenario calculamos el profit en cada precio.
   5. Comparamos cada precio contra un ORÁCULO (el mejor profit posible
      en ese escenario, visto en retrospectiva) → eso define el REGRET.
   6. Elegimos precio bajo dos criterios:
        · regret ESPERADO   → precio robusto "en promedio"  (p_g*)
        · regret PEOR CASO  → precio robusto "minimax"       (p_g_worst)
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt

# Formato de pesos chilenos para los ejes (se reutiliza en todos los gráficos)
clp = plt.FuncFormatter(lambda x, _: f"${x:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════

# --- Demanda: nivel base (α) ---
demanda_base        = 200      # α̂  : demanda esperada (estimación puntual)
demanda_base_sigma  = 20       # σ_α: incertidumbre sobre el nivel

# --- Demanda: sensibilidad al precio (α_p) ---
sensib_precio       = 0.008    # α̂_p : cuánto reacciona la demanda al valor percibido
sensib_precio_sigma = 0.003    # σ_αp: incertidumbre sobre la pendiente
                               #       (clave: solo si es > 0, el minimax se separa del esperado)

# --- Aversión a la pérdida (Kahneman-Tversky) ---
lam   = 2.25       # λ : las pérdidas pesan 2.25× más que las ganancias (valor empírico, NO se estima)
r_ref = 12_000     # r : precio de referencia del cliente (ancla de la asimetría)

# --- Costo y grilla de precios ---
costo  = 3_000                     # costo unitario (CLP, fijo y conocido)
P_MIN, P_MAX, N_PRECIOS = 5_000, 20_000, 50

# --- Simulación ---
N_ESCENARIOS = 1_000
SEED         = 42                  # semilla fija → resultados reproducibles


# ═══════════════════════════════════════════════════════════════════════════
# 2. GRILLA DE PRECIOS  (los precios candidatos entre los que elegimos)
# ═══════════════════════════════════════════════════════════════════════════
precios = np.linspace(P_MIN, P_MAX, N_PRECIOS)          # shape (N_PRECIOS,)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ESCENARIOS DE INCERTIDUMBRE
#    Cada escenario es un par (α_k, α_p,k) muestreado de sus distribuciones.
# ═══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)

alpha    = rng.normal(demanda_base,  demanda_base_sigma,  N_ESCENARIOS)   # nivel por escenario
alpha_p  = rng.normal(sensib_precio, sensib_precio_sigma, N_ESCENARIOS)   # pendiente por escenario
alpha_p  = np.clip(alpha_p, 1e-6, None)                                   # la sensibilidad no puede ser negativa


# ═══════════════════════════════════════════════════════════════════════════
# 4. UTILIDAD TRANSACCIONAL DE KAHNEMAN
#    v(p) = [r − p]⁺  −  λ·[p − r]⁺
#      · si p < r → "ganancia" percibida, pendiente +1
#      · si p > r → "pérdida"  percibida, pendiente −λ  (penalización amplificada)
# ═══════════════════════════════════════════════════════════════════════════
def valor_kahneman(p, r=r_ref, lam=lam):
    ganancia = np.maximum(0, r - p)        # cuánto "gana" el cliente si el precio está bajo r
    perdida  = np.maximum(0, p - r)        # cuánto "pierde" si está sobre r
    return ganancia - lam * perdida

valor = valor_kahneman(precios)            # shape (N_PRECIOS,)


# ═══════════════════════════════════════════════════════════════════════════
# 5. DEMANDA POR ESCENARIO
#    D_k(p) = max(0,  α_k  +  α_p,k · v(p))
#    Matriz (N_ESCENARIOS × N_PRECIOS): demanda de cada escenario en cada precio.
# ═══════════════════════════════════════════════════════════════════════════
demanda = np.maximum(0, alpha[:, None] + alpha_p[:, None] * valor[None, :])


# ═══════════════════════════════════════════════════════════════════════════
# 6. PROFIT POR ESCENARIO
#    Π_k(p) = D_k(p) · (p − costo)
# ═══════════════════════════════════════════════════════════════════════════
profit = demanda * (precios[None, :] - costo)          # shape (N_ESCENARIOS × N_PRECIOS)


# ═══════════════════════════════════════════════════════════════════════════
# 7. REGRET Y CRITERIOS DE DECISIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Oráculo: el mejor profit alcanzable en cada escenario (visto en retrospectiva).
profit_oraculo = profit.max(axis=1)                    # shape (N_ESCENARIOS,)

# Regret[k, j] = lo que dejamos de ganar en el escenario k al cobrar el precio j.
regret = profit_oraculo[:, None] - profit              # shape (N_ESCENARIOS × N_PRECIOS), siempre ≥ 0

# Resumen del regret por precio (promediando / tomando extremos sobre escenarios):
regret_esperado = regret.mean(axis=0)                  # E[R] por precio
regret_peor      = regret.max(axis=0)                  # peor caso por precio (minimax)
regret_mejor     = regret.min(axis=0)                  # mejor caso por precio (solo como cota inferior)

# --- Criterio A: maximizar profit esperado ---
p_profit_esperado = precios[np.argmax(profit.mean(axis=0))]

# --- Criterio B: minimizar regret esperado ---
p_regret_esperado = precios[np.argmin(regret_esperado)]

# --- Criterio C: minimizar regret de peor caso (minimax) ---
p_regret_minimax  = precios[np.argmin(regret_peor)]

# Identidad clave de la tesis: A y B coinciden SIEMPRE (el oráculo cancela
# como constante en el regret esperado → minimizar E[R] ≡ maximizar E[Π]).
print(f"Criterio A — max profit esperado   : ${p_profit_esperado:,.0f}")
print(f"Criterio B — min regret esperado   : ${p_regret_esperado:,.0f}   (= A por construcción)")
print(f"Criterio C — min regret peor caso  : ${p_regret_minimax:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

# ── GRÁFICO 1 · Curvas de demanda ───────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 6))
for k in range(50):                                    # mostramos 50 de los 1.000 escenarios
    ax.plot(precios, demanda[k], lw=0.8, alpha=0.4, color="#60a5fa")
ax.plot(precios, valor_kahneman(precios) * sensib_precio + demanda_base,
        lw=2.5, color="#f59e0b", label="demanda con parámetros estimados (α̂, α̂_p)")
ax.axvline(r_ref, color="#a78bfa", ls="--", lw=1.5,
           label=f"precio de referencia  r = ${r_ref:,.0f}")
ax.set(title="1 · Demanda con aversión a la pérdida  (50 de 1.000 escenarios)",
       xlabel="Precio (CLP)", ylabel="Demanda (unidades)")
ax.legend(); ax.grid(ls="--", lw=0.4, alpha=0.4); ax.xaxis.set_major_formatter(clp)
fig1.tight_layout()

# ── GRÁFICO 2 · Regret por precio (con marco de riesgo mejor–peor caso) ──────
fig2, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(precios, regret_mejor, regret_peor, color="#fb923c", alpha=0.13,
                label="rango mejor–peor caso (marco de riesgo)")
ax.plot(precios, regret_peor,  lw=1.8, color="#fb923c",            label="peor caso  (minimax)")
ax.plot(precios, regret_mejor, lw=1.8, color="#fb923c", ls=":",    label="mejor caso (cota inferior)")
ax.plot(precios, regret_esperado, lw=2.6, color="#dc2626",         label="regret esperado")
ax.axvline(p_regret_esperado, color="gold",    ls="--", lw=1.6,
           label=f"p_g* = ${p_regret_esperado:,.0f}  (min regret esperado)")
ax.axvline(p_regret_minimax,  color="#16a34a", ls="--", lw=1.6,
           label=f"p_g_worst = ${p_regret_minimax:,.0f}  (min regret peor caso)")
ax.set(title="2 · Regret por precio: esperado dentro del marco mejor–peor caso",
       xlabel="Precio (CLP)", ylabel="Regret (CLP)", ylim=(0, None))
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
ax.grid(ls="--", lw=0.4, alpha=0.4)
ax.xaxis.set_major_formatter(clp); ax.yaxis.set_major_formatter(clp)
fig2.tight_layout()

# ── GRÁFICO 3 · Histogramas: ¿qué pasa al fijar cada precio óptimo? ──────────
# Fijamos el precio y miramos cómo se distribuyen los 1.000 escenarios en él.
j_esperado = np.argmin(regret_esperado)
j_minimax  = np.argmin(regret_peor)

demanda_en_esperado, demanda_en_minimax = demanda[:, j_esperado], demanda[:, j_minimax]
profit_en_esperado,  profit_en_minimax  = profit[:,  j_esperado], profit[:,  j_minimax]

fig3, (axD, axP) = plt.subplots(1, 2, figsize=(14, 5.5))

axD.hist(demanda_en_esperado, bins=40, alpha=0.55, color="#60a5fa",
         label=f"en p_g* = ${p_regret_esperado:,.0f}")
axD.hist(demanda_en_minimax,  bins=40, alpha=0.55, color="#34d399",
         label=f"en p_g_worst = ${p_regret_minimax:,.0f}")
axD.set(title="3 · Distribución de la DEMANDA al fijar cada precio óptimo",
        xlabel="Demanda (unidades)", ylabel="N° de escenarios")
axD.legend(fontsize=9); axD.grid(ls="--", lw=0.4, alpha=0.4)

axP.hist(profit_en_esperado, bins=40, alpha=0.55, color="#f87171",
         label=f"en p_g* = ${p_regret_esperado:,.0f}")
axP.hist(profit_en_minimax,  bins=40, alpha=0.55, color="#fbbf24",
         label=f"en p_g_worst = ${p_regret_minimax:,.0f}")
axP.set(title="Distribución del PROFIT al fijar cada precio óptimo",
        xlabel="Profit (CLP)", ylabel="N° de escenarios")
axP.legend(fontsize=9); axP.grid(ls="--", lw=0.4, alpha=0.4)
axP.xaxis.set_major_formatter(clp)
fig3.tight_layout()

# ── Estadísticos del corte (para el informe) ────────────────────────────────
def resumen(nombre, x, en_clp=False):
    f = (lambda v: f"${v:,.0f}") if en_clp else (lambda v: f"{v:,.1f}")
    print(f"  {nombre:22s} media={f(x.mean())}  σ={f(x.std())}  "
          f"P5={f(np.percentile(x,5))}  P95={f(np.percentile(x,95))}")

print("\n── Distribución en el corte de cada precio óptimo ──")
resumen("Demanda @ p_g*",      demanda_en_esperado)
resumen("Demanda @ p_g_worst", demanda_en_minimax)
resumen("Profit  @ p_g*",      profit_en_esperado,  en_clp=True)
resumen("Profit  @ p_g_worst", profit_en_minimax,   en_clp=True)

# Muestra los tres gráficos en pantalla.
plt.show()