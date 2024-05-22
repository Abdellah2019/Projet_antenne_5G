import numpy as np
import matplotlib.pyplot as plt

# Paramètres d'entrée (exemples, à adapter selon votre cas)
phi_3dB = 10  # Largeur de bande horizontale de 3dB en degrés
theta_3dB = 10  # Largeur de bande verticale de 3dB en degrés
Am = 25  # Rapport avant/arrière de l'antenne en dB
SLA_v = 30  # Niveau de rayonnement secondaire en dB
G_Emax = 10  # Gain maximal de l'élément unique en dB

# Convertir les angles en radians pour les calculs
phi_3dB_rad = np.deg2rad(phi_3dB)
theta_3dB_rad = np.deg2rad(theta_3dB)

# Fonction pour le diagramme d'antenne horizontal
def A_EH(phi):
    phi_rad = np.deg2rad(phi)
    return -np.minimum(12 * (phi_rad / phi_3dB_rad)**2, Am)

# Fonction pour le diagramme de rayonnement vertical
def A_EV(theta):
    theta_rad = np.deg2rad(theta)
    return -np.minimum(12 * ((theta_rad - np.pi / 2) / theta_3dB_rad)**2, SLA_v)

# Fonction pour le diagramme d'un élément unique
def A_E(phi, theta):
    return G_Emax - np.minimum(-(A_EH(phi) + A_EV(theta)), Am)

# Génération des données pour les angles
phi_values = np.linspace(-180, 180, 360)
theta_values = np.linspace(0, 180, 180)

# Calculer le diagramme d'antenne horizontal et vertical
A_EH_values = A_EH(phi_values)
A_EV_values = A_EV(theta_values)

# Affichage des résultats
plt.figure(figsize=(14, 6))

# Diagramme d'antenne horizontal
plt.subplot(1, 2, 1)
plt.plot(phi_values, A_EH_values)
plt.title("Diagramme d'antenne horizontal")
plt.xlabel("Angle φ (degrés)")
plt.ylabel("Gain (dB)")
plt.grid(True)

# Diagramme de rayonnement vertical
plt.subplot(1, 2, 2)
plt.plot(theta_values, A_EV_values)
plt.title("Diagramme de rayonnement vertical")
plt.xlabel("Angle θ (degrés)")
plt.ylabel("Gain (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()
