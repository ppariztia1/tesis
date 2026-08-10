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

╔═════════════════════════════════════════════════════════════════════════════╗
║     PARA DOCUMENTAR EN LA TESIS  (checklist — no se me puede pasar)         ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  [D1] SUPUESTO DEL ORÁCULO. El oráculo es el máximo profit por escenario,   ║
║       visto en retrospectiva (clairvoyant benchmark). Como trabajo con      ║
║       datos SIMULADOS, no existe un "caso real" observable: el oráculo usa  ║
║       el mejor resultado posible de cada escenario como proxy. Declararlo   ║
║       como LIMITACIÓN explícita. (Ítem 3 de la reunión del 11-jun.)         ║
║                                                                             ║
║  [D2] RESOLUCIÓN DE GRILLA. Reportar N_PRECIOS y el paso en CLP. Los        ║
║       precios óptimos tienen precisión LIMITADA por la grilla; no reportar  ║
║       más cifras significativas que las que la grilla permite.              ║
║                                                                             ║
║  [D3] SEMILLA FIJA (SEED). Reportarla para reproducibilidad. El profe lo    ║
║       avaló como buena práctica.                                            ║
║                                                                             ║
║  [D4] FRAGILIDAD DEL MINIMAX. Solo ~5–7 escenarios de 1.000 "atan" la curva ║
║       de peor caso; uno casi-plano (creado por el clip de α_p, ver [D5])    ║
║       domina. p_g_worst es ~2× más sensible a la semilla que p_g*.          ║
║       Presentarlo como PROPIEDAD del método de peor caso, no esconderlo.    ║
║                                                                             ║
║  [D5] CLIP DE α_p. El piso np.clip(α_p, 1e-6, None) evita sensibilidad      ║
║       negativa, pero genera el escenario casi-plano que domina el minimax.  ║
║       Es la limitación concreta detrás de [D4].                             ║
║                                                                             ║
║  [D6] IDENTIDAD p_g* = p_EV. Es ALGEBRAICA, no coincidencia: el oráculo     ║
║       cancela como constante en el regret esperado → minimizar E[R] ≡       ║
║       maximizar E[Π]. CONSECUENCIA: sin el criterio minimax, el marco de    ║
║       regret colapsa a maximización de profit esperado. El minimax es lo    ║
║       que hace que el regret aporte algo metodológicamente distinto.        ║
║                                                                             ║
║  [D7] DOS AVERSIONES DISTINTAS (no confundirlas bajo "aversión al riesgo"): ║
║       (a) λ=2.25 = aversión a la PÉRDIDA, efecto prospect theory del lado   ║
║           del CONSUMIDOR, embebido en cada curva de demanda.                ║
║       (b) minimax = robustez de peor caso, criterio del lado de la FIRMA,   ║
║           entre escenarios.                                                 ║
║                                                                             ║
║  [D8] MEJOR CASO = cota inferior de la banda, NO criterio de decisión (por  ║
║       eso no lleva línea vertical propia). Es trivialmente optimista (≈0    ║
║       en casi todo el rango).                                               ║
║                                                                             ║
║  [D9] SEPARACIÓN p_g_worst ≠ p_g*. Solo aparece si hay incertidumbre en la  ║
║       PENDIENTE (sensib_precio_sigma > 0). Con σ_αp = 0 ambos coinciden.    ║
║                                                                             ║
║  [D10] λ = 2.25 es valor EMPÍRICO de Kahneman-Tversky; NO se estima.        ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt

# Formato CLP
clp = plt.FuncFormatter(lambda x, _: f"${x:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════

# Demanda: nivel base (α)
demanda_base        = 200
demanda_base_sigma  = 20

# Demanda: sensibilidad al precio (α_p)
sensib_precio       = 0.008
sensib_precio_sigma = 0.003
#       [D9] si esto fuera 0, p_g_worst = p_g*. La separación
#            del minimax depende de esta incertidumbre.

# Aversión a la pérdida
lam   = 2.25       # [D10] λ empírico: las pérdidas pesan 2.25× más que las ganancias. NO se estima.
r_ref = 12_000 

# Costo y grilla de precios
costo  = 3_000

# [D2] Resolución de grilla: subimos de 50 → 500 puntos.
#      REPORTAR en el doc: N_PRECIOS y el paso resultante.
P_MIN, P_MAX, N_PRECIOS = 5_000, 20_000, 500

# Simulación
N_ESCENARIOS = 1_000
SEED = 42            # [D3] semilla fija → reproducibilidad (reportarla)

# ═══════════════════════════════════════════════════════════════════════════
# 2. GRILLA DE PRECIOS
# ═══════════════════════════════════════════════════════════════════════════
precios = np.linspace(P_MIN, P_MAX, N_PRECIOS)
paso_grilla = precios[1] - precios[0]

# ═══════════════════════════════════════════════════════════════════════════
# 3. ESCENARIOS DE INCERTIDUMBRE
# ═══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)

alpha    = rng.normal(demanda_base,  demanda_base_sigma,  N_ESCENARIOS)
alpha_p  = rng.normal(sensib_precio, sensib_precio_sigma, N_ESCENARIOS)
# [D5] Este clip evita pendiente negativa, pero crea el escenario casi-plano
#      que domina el minimax. Documentar como limitación.
alpha_p  = np.clip(alpha_p, 1e-6, None)

# ═══════════════════════════════════════════════════════════════════════════
# 4. UTILIDAD TRANSACCIONAL DE KAHNEMAN
# ═══════════════════════════════════════════════════════════════════════════
def valor_kahneman(p, r=r_ref, lam=lam):
    ganancia = np.maximum(0, r - p)
    perdida  = np.maximum(0, p - r)
    return ganancia - lam * perdida

# ═══════════════════════════════════════════════════════════════════════════
# 5-7. PIPELINE DEL MODELO  (demanda → profit → regret → criterios)
# ═══════════════════════════════════════════════════════════════════════════
def evaluar_modelo(lam_eval, alpha, alpha_p, precios):
    valor   = valor_kahneman(precios, lam=lam_eval)

    # 5. DEMANDA POR ESCENARIO  D_k(p) = max(0, α_k + α_p,k · v(p))
    demanda = np.maximum(0, alpha[:, None] + alpha_p[:, None] * valor[None, :])

    # 6. PROFIT POR ESCENARIO   Π_k(p) = D_k(p) · (p − costo)
    profit  = demanda * (precios[None, :] - costo)

    # 7. REGRET Y CRITERIOS
    # [D1] ORÁCULO: mejor profit por escenario, en retrospectiva. Proxy del
    #      "caso real" porque los datos son simulados. Documentar como supuesto.
    profit_oraculo = profit.max(axis=1)
    regret = profit_oraculo[:, None] - profit

    regret_esperado = regret.mean(axis=0)     # E[R] por precio
    regret_peor     = regret.max(axis=0)      # minimax
    regret_mejor    = regret.min(axis=0)      # [D8] mejor caso: SOLO cota inferior

    # Criterios de decisión (precios):
    p_profit_esperado = precios[np.argmax(profit.mean(axis=0))]   # A: max E[Π]
    p_regret_esperado = precios[np.argmin(regret_esperado)]       # B: min E[R]  (= A por [D6])
    p_regret_minimax  = precios[np.argmin(regret_peor)]           # C: min peor caso

    return dict(valor=valor, demanda=demanda, profit=profit, regret=regret,
                regret_esperado=regret_esperado, regret_peor=regret_peor,
                regret_mejor=regret_mejor, profit_oraculo=profit_oraculo,
                p_profit_esperado=p_profit_esperado,
                p_regret_esperado=p_regret_esperado,
                p_regret_minimax=p_regret_minimax)

# Corremos el MODELO PRINCIPAL
m = evaluar_modelo(lam, alpha, alpha_p, precios)

demanda, profit, regret = m["demanda"], m["profit"], m["regret"]
regret_esperado, regret_peor, regret_mejor = m["regret_esperado"], m["regret_peor"], m["regret_mejor"]
p_profit_esperado = m["p_profit_esperado"]
p_regret_esperado = m["p_regret_esperado"]
p_regret_minimax  = m["p_regret_minimax"]

# [D6] Identidad clave: A y B coinciden SIEMPRE (el oráculo cancela como
#      constante en el regret esperado → minimizar E[R] = maximizar E[Π]).
print(f"Grilla: {N_PRECIOS} precios, paso = ${paso_grilla:,.0f}  (semilla = {SEED})")
print(f"Criterio A — max profit esperado   : ${p_profit_esperado:,.0f}")
print(f"Criterio B — min regret esperado   : ${p_regret_esperado:,.0f}   (= A por construcción)")
print(f"Criterio C — min regret peor caso  : ${p_regret_minimax:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════

# ── GRÁFICO 1: Curvas de demanda ───────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 6))
for k in range(50):
    ax.plot(precios, demanda[k], lw=0.8, alpha=0.4, color="#60a5fa")

demanda_estimada = np.maximum(0, demanda_base + sensib_precio * valor_kahneman(precios))
ax.plot(precios, demanda_estimada, lw=2.5, color="#f59e0b",
        label="demanda con parámetros estimados (α̂, α̂_p)")
ax.axvline(r_ref, color="#a78bfa", ls="--", lw=1.5,
           label=f"precio de referencia  r = ${r_ref:,.0f}")
ax.set(title="1 · Demanda con aversión a la pérdida  (50 de 1.000 escenarios)",
       xlabel="Precio (CLP)", ylabel="Demanda (unidades)")
ax.legend(); ax.grid(ls="--", lw=0.4, alpha=0.4); ax.xaxis.set_major_formatter(clp)
fig1.tight_layout()

# ── GRÁFICO 2: Profit por precio ───────────────────────────────────────────
#   [D6]: la vertical en p_g* cae exactamente sobre el máximo del profit esperado

fig2, ax = plt.subplots(figsize=(10, 6))
for k in range(50):
    ax.plot(precios, profit[k], lw=0.8, alpha=0.4, color="#fca5a5")
ax.plot(precios, profit.mean(axis=0), lw=2.6, color="#b91c1c",
        label="profit esperado (promedio de escenarios)")
ax.axvline(p_regret_esperado, color="gold", ls="--", lw=1.6,
           label=f"p_g* = ${p_regret_esperado:,.0f}  (máx profit esperado = p_EV)")
ax.axvline(r_ref, color="#a78bfa", ls="--", lw=1.2,
           label=f"precio de referencia  r = ${r_ref:,.0f}")
ax.set(title="2 · Profit por precio  (50 de 1.000 escenarios)",
       xlabel="Precio (CLP)", ylabel="Profit (CLP)")
ax.legend(fontsize=9); ax.grid(ls="--", lw=0.4, alpha=0.4)
ax.xaxis.set_major_formatter(clp); ax.yaxis.set_major_formatter(clp)
fig2.tight_layout()

# ── GRÁFICO 3: Regret por precio (con marco de riesgo mejor–peor caso) ──────
fig3, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(precios, regret_mejor, regret_peor, color="#fb923c", alpha=0.13,
                label="rango mejor–peor caso (marco de riesgo)")
ax.plot(precios, regret_peor,  lw=1.8, color="#fb923c",            label="peor caso  (minimax)")
ax.plot(precios, regret_mejor, lw=1.8, color="#fb923c", ls=":",    label="mejor caso (cota inferior)")
ax.plot(precios, regret_esperado, lw=2.6, color="#dc2626",         label="regret esperado")
ax.axvline(p_regret_esperado, color="gold",    ls="--", lw=1.6,
           label=f"p_g* = ${p_regret_esperado:,.0f}  (min regret esperado)")
ax.axvline(p_regret_minimax,  color="#16a34a", ls="--", lw=1.6,
           label=f"p_g_worst = ${p_regret_minimax:,.0f}  (min regret peor caso)")
# [D8] OJO: NO agregar una línea vertical para el mejor caso. Es cota, no criterio.
ax.set(title="3 · Regret por precio: esperado dentro del marco mejor–peor caso",
       xlabel="Precio (CLP)", ylabel="Regret (CLP)", ylim=(0, None))
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
ax.grid(ls="--", lw=0.4, alpha=0.4)
ax.xaxis.set_major_formatter(clp); ax.yaxis.set_major_formatter(clp)
fig3.tight_layout()

# ── GRÁFICO 4: Histogramas: efecto de la AVERSIÓN A LA PÉRDIDA ──────────────
m_sim  = evaluar_modelo(1.0, alpha, alpha_p, precios)
m_asim = m

print(f"\n── Gráfico 4 · efecto de β (aversión a la pérdida) ──")
print(f"   p_g* con λ=1   (simétrico) = ${m_sim['p_regret_esperado']:,.0f}   "
      f"(solución de ESQUINA: pega en el tope de la grilla)")
print(f"   p_g* con λ=2.25 (asimétr.) = ${m_asim['p_regret_esperado']:,.0f}   "
      f"(óptimo INTERIOR, creado por la aversión a la pérdida)")

p_corte = m_asim["p_regret_esperado"]
j_corte = int(np.argmin(m_asim["regret_esperado"]))

dem_sim,  dem_asim  = m_sim["demanda"][:, j_corte],  m_asim["demanda"][:, j_corte]
prof_sim, prof_asim = m_sim["profit"][:,  j_corte],  m_asim["profit"][:,  j_corte]

fig4, (axD, axP) = plt.subplots(1, 2, figsize=(14, 5.5))

axD.hist(dem_sim,  bins=40, alpha=0.55, color="#94a3b8", label="β simétrico  (λ=1)")
axD.hist(dem_asim, bins=40, alpha=0.55, color="#6366f1", label="β asimétrico (λ=2.25)")
axD.set(title=f"4. DEMANDA al cobrar p_g* = ${p_corte:,.0f}\n(β simétrico vs asimétrico — efecto de la aversión a la pérdida)",
        xlabel="Demanda (unidades)", ylabel="N° de escenarios")
axD.legend(fontsize=9); axD.grid(ls="--", lw=0.4, alpha=0.4)

axP.hist(prof_sim,  bins=40, alpha=0.55, color="#94a3b8", label="β simétrico  (λ=1)")
axP.hist(prof_asim, bins=40, alpha=0.55, color="#dc2626", label="β asimétrico (λ=2.25)")
axP.set(title=f"PROFIT al cobrar p_g* = ${p_corte:,.0f}\n(β simétrico vs asimétrico)",
        xlabel="Profit (CLP)", ylabel="N° de escenarios")
axP.legend(fontsize=9); axP.grid(ls="--", lw=0.4, alpha=0.4)
axP.xaxis.set_major_formatter(clp)
fig4.tight_layout()

# ── Estadísticos del corte (para el informe) ────────────────────────────────
def resumen(nombre, x, en_clp=False):
    f = (lambda v: f"${v:,.0f}") if en_clp else (lambda v: f"{v:,.1f}")
    print(f"  {nombre:28s} media={f(x.mean())}  σ={f(x.std())}  "
          f"P5={f(np.percentile(x,5))}  P95={f(np.percentile(x,95))}")

print(f"\n   (comparación a precio común p_g* = ${p_corte:,.0f})")
resumen("Demanda  β simétrico (λ=1)",    dem_sim)
resumen("Demanda  β asimétrico (λ=2.25)", dem_asim)
resumen("Profit   β simétrico (λ=1)",    prof_sim,  en_clp=True)
resumen("Profit   β asimétrico (λ=2.25)", prof_asim, en_clp=True)

plt.show()