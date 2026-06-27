import numpy as np

# a) Données
x = np.array([20,35,45,50,60,70,80,90,100,120])
y = np.array([55000,82000,105000,115000,135000,152000,178000,195000,220000,260000])
n = len(x)

# b) Moyennes
x_bar = np.mean(x)
y_bar = np.mean(y)

# c) Estimateur MCO
a_star = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar)**2)
b_star = y_bar - a_star * x_bar

# d) Prédictions et LMSE optimal
y_hat = a_star * x + b_star
LMSE_star = np.mean((y - y_hat)**2)

# Modèle naïf de référence
a0, b0 = 1800, 28900
LMSE0 = np.mean((y - (a0*x + b0))**2)
gain = (LMSE0 - LMSE_star) / LMSE0 * 100

# e) Affichage
print(f"a* = {a_star:.2f} DH/m2")
print(f"b* = {b_star:.2f} DH")
print(f"LMSE(theta*) = {LMSE_star:.2f}")
print(f"Gain = {gain:.2f} %")
# verifier le gradient
residus = y - y_hat

dL_da = -2/n * np.sum(x * residus)
dL_db = -2/n * np.sum(residus)

print(f"dL/da(theta*) = {dL_da:.2e}")
print(f"dL/db(theta*) = {dL_db:.2e}")
print("Condition verifiee ?", abs(dL_da) < 1e-4 and abs(dL_db) < 1e-4)

# tracer les deux droites
import matplotlib.pyplot as plt

x_line = np.linspace(x.min(), x.max(), 200)
y0_line = a0 * x_line + b0
ystar_line = a_star * x_line + b_star

plt.figure(figsize=(7,5))
plt.scatter(x, y, color='black', label='Observations')
plt.plot(x_line, y0_line, 'k--', label=f'Droite initiale (a0={a0}, b0={b0})')
plt.plot(x_line, ystar_line, 'b-', label=f'Droite optimale (a*={a_star:.2f}, b*={b_star:.2f})')
plt.xlabel('Superficie x (m²)')
plt.ylabel('Prix y (DH)')
plt.title('Nuage de points et droites de régression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('droites.pdf')
plt.show()
#analyse des résidus
res0 = y - (a0 * x + b0)
res_star = y - y_hat
idx = np.arange(1, n+1)

plt.figure(figsize=(7,5))
plt.plot(idx, res0, 'k--', marker='o', label='Résidus modèle initial')
plt.plot(idx, res_star, 'b-', marker='o', label='Résidus modèle optimal')
plt.axhline(0, color='red', linestyle='--', label='Référence r = 0')
plt.xlabel('Indice i')
plt.ylabel('Résidu (DH)')
plt.title('Résidus des deux modèles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('residus.pdf')
plt.show()