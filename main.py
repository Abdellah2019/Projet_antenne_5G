import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Fonction pour le diagramme d'antenne horizontal
def A_EH(phi, phi_3dB_rad, Am):
    phi_rad = np.deg2rad(phi)
    return -np.minimum(12 * (phi_rad / phi_3dB_rad)**2, Am)

# Fonction pour le diagramme de rayonnement vertical
def A_EV(theta, theta_3dB_rad, SLA_v):
    theta_rad = np.deg2rad(theta)
    return -np.minimum(12 * ((theta_rad - np.pi / 2) / theta_3dB_rad)**2, SLA_v)

# Fonction pour le diagramme d'un élément unique
def A_E(phi, theta, G_Emax, phi_3dB_rad, theta_3dB_rad, Am, SLA_v):
    return G_Emax - np.minimum(-(A_EH(phi, phi_3dB_rad, Am) + A_EV(theta, theta_3dB_rad, SLA_v)), Am)

# Configuration de l'interface utilisateur Streamlit
st.title("Simulation d'une antenne active")

# Saisie des paramètres par l'utilisateur
phi_3dB = st.slider("Largeur de bande horizontale de 3dB (degrés)", min_value=1, max_value=180, value=10)
theta_3dB = st.slider("Largeur de bande verticale de 3dB (degrés)", min_value=1, max_value=180, value=10)
Am = st.slider("Rapport avant/arrière de l'antenne (dB)", min_value=1, max_value=50, value=25)
SLA_v = st.slider("Niveau de rayonnement secondaire (dB)", min_value=1, max_value=50, value=30)
G_Emax = st.slider("Gain maximal de l'élément unique (dB)", min_value=1, max_value=50, value=10)

# Convertir les angles en radians pour les calculs
phi_3dB_rad = np.deg2rad(phi_3dB)
theta_3dB_rad = np.deg2rad(theta_3dB)

# Génération des données pour les angles
phi_values = np.linspace(-180, 180, 360)
theta_values = np.linspace(0, 180, 180)

# Calculer le diagramme d'antenne horizontal et vertical
A_EH_values = A_EH(phi_values, phi_3dB_rad, Am)
A_EV_values = A_EV(theta_values, theta_3dB_rad, SLA_v)

# Génération des données pour le diagramme 3D
phi_3d = np.linspace(-180, 180, 360)
theta_3d = np.linspace(0, 180, 180)
phi_3d, theta_3d = np.meshgrid(phi_3d, theta_3d)
A_E_values = A_E(phi_3d, theta_3d, G_Emax, phi_3dB_rad, theta_3dB_rad, Am, SLA_v)

# Affichage des résultats en 2D
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Diagramme d'antenne horizontal
ax[0].plot(phi_values, A_EH_values)
ax[0].set_title("Diagramme d'antenne horizontal")
ax[0].set_xlabel("Angle φ (degrés)")
ax[0].set_ylabel("Gain (dB)")
ax[0].grid(True)

# Diagramme de rayonnement vertical
ax[1].plot(theta_values, A_EV_values)
ax[1].set_title("Diagramme de rayonnement vertical")
ax[1].set_xlabel("Angle θ (degrés)")
ax[1].set_ylabel("Gain (dB)")
ax[1].grid(True)

st.pyplot(fig)

# Affichage du diagramme 3D
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
phi_3d_rad = np.deg2rad(phi_3d)
theta_3d_rad = np.deg2rad(theta_3d)

# Coordonnées sphériques vers cartésiennes
X = A_E_values * np.sin(theta_3d_rad) * np.cos(phi_3d_rad)
Y = A_E_values * np.sin(theta_3d_rad) * np.sin(phi_3d_rad)
Z = A_E_values * np.cos(theta_3d_rad)

ax_3d.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='viridis', edgecolor='none')
ax_3d.set_title("Diagramme 3D de l'antenne")
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")

st.pyplot(fig_3d)
