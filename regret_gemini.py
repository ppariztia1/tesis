import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('MacOSX') # Esto obliga a usar la ventana nativa de Mac
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN BÁSICA
costo = 10
precios_posibles = np.linspace(15, 50, 100) # 100 puntos para que se vea suave
cuantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
colores = {0.1: 'red', 0.3: 'orange', 0.5: 'gray', 0.7: 'blue', 0.9: 'green'}

def predecir_demanda(cuantil, precio):
    base_por_cuantil = {0.1: 60, 0.3: 80, 0.5: 100, 0.7: 120, 0.9: 140}
    demanda = base_por_cuantil[cuantil] - 2 * precio
    return max(0, demanda)

# 2. CÁLCULO DE REGLAS Y REGRET
precios_elegidos = {}
for g in cuantiles:
    profits_imaginarios = [predecir_demanda(g, p) * (p - costo) for p in precios_posibles]
    precios_elegidos[g] = precios_posibles[np.argmax(profits_imaginarios)]

best_profit_s = {s: max([predecir_demanda(s, p) * (p - costo) for p in precios_posibles]) for s in cuantiles}

resultados_regret = []
for g in cuantiles:
    p_g = precios_elegidos[g]
    regrets = [best_profit_s[s] - (predecir_demanda(s, p_g) * (p_g - costo)) for s in cuantiles]
    resultados_regret.append({'Regla (g)': g, 'Precio': p_g, 'Regret Promedio': np.mean(regrets)})

# 3. GENERAR EL GRÁFICO (Aquí estaba el detalle)
plt.figure(figsize=(10, 6))

# Dibujar cada curva de demanda
for q in cuantiles:
    demandas = [predecir_demanda(q, p) for p in precios_posibles]
    plt.plot(precios_posibles, demandas, label=f'Escenario q={q}', color=colores[q], lw=2)
    
    # Marcar el precio que elegiría cada regla sobre su propia curva
    p_g = precios_elegidos[q]
    plt.scatter(p_g, predecir_demanda(q, p_g), color=colores[q], edgecolor='black', s=100, zorder=5)

plt.title('Visualización de Escenarios de Demanda', fontsize=14)
plt.xlabel('Precio ($p$)', fontsize=12)
plt.ylabel('Demanda ($d$)', fontsize=12)
plt.axvline(x=costo, color='black', linestyle='--', label=f'Costo (c={costo})')
plt.grid(True, alpha=0.3)
plt.legend()

# 4. MOSTRAR RESULTADOS
df = pd.DataFrame(resultados_regret)
print("--- TABLA DE REGRET ---")
print(df.round(2).to_string(index=False))
print(f"\n🏆 MEJOR REGLA: {df.loc[df['Regret Promedio'].idxmin()]['Regla (g)']}")

plt.show()