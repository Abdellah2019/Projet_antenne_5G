import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D


st.set_page_config(page_title='Antenne Active', page_icon=None,layout = 'wide', initial_sidebar_state = 'auto')

# Fonction pour le diagramme d'antenne horizontal
def A_EH(phi, phi_3dB_rad, Am,offset):
    phi_rad = np.deg2rad(phi)
    return offset-np.minimum(12 * (phi_rad / phi_3dB_rad)**2, Am)

# Fonction pour le diagramme de rayonnement vertical
def A_EV(theta, theta_3dB_rad, SLA_v,offset):
    theta_rad = np.deg2rad(theta)
    return offset-np.minimum(12 * ((theta_rad - np.pi / 2) / theta_3dB_rad)**2, SLA_v)

# Fonction pour le diagramme d'un élément unique
def A_E(phi, theta, G_Emax, phi_3dB_rad, theta_3dB_rad, Am, SLA_v,offset):
    return G_Emax +offset - np.minimum(-(A_EH(phi, phi_3dB_rad, Am,offset) + A_EV(theta, theta_3dB_rad, SLA_v,offset)), Am)


# Fonction pour le vecteur de superposition
def v_nm(n, m, theta, phi, d_v, d_h, lam):
    return np.exp(-1 * 2 * np.pi * ((n-1) * d_v / lam * np.cos(theta) + (m-1) * d_h / lam * np.sin(theta) * np.sin(phi)))

# Fonction pour la pondération
def w_inm(n, m, theta_i_max, phi_i_max, d_v, d_h, lam, N_v, N_h):
    return (1 / np.sqrt((N_v * N_h))) * np.exp(-1 * 2 * np.pi * ((n-1) * d_v / lam * np.sin(theta_i_max) - (m-1) * d_h / lam * np.cos(theta_i_max) * np.sin(phi_i_max)))


# Fonction pour le diagramme composite 
def A_Beam(theta, phi, A_e, N_v, N_h, d_v, d_h, lam, theta_i_max, phi_i_max):
    
    v = np.zeros((N_v, N_h), dtype=complex)
    w = np.zeros((N_v, N_h), dtype=complex)

    for n in range(1, N_v+1):
        for m in range(1, N_h+1):
            v[n-1, m-1] = v_nm(n, m, theta, phi, d_v, d_h, lam)
            w[n-1, m-1] = w_inm(n, m, theta_i_max, phi_i_max, d_v, d_h, lam, N_v, N_h)
    
    return A_e + 10 * np.log10(np.abs(np.sum(w * v, axis=(0, 1)))**2)



# Configuration de l'interface utilisateur Streamlit
st.title("Simulation d'une antenne active 5G")


tab1, tab2= st.tabs(["Diagramme d'un element unique ", "Diagramme d'antenne Composite"])

with tab1:
    # Saisie des paramètres par l'utilisateur
    phi_3dB = st.slider("Largeur de bande horizontale de 3dB (degrés)", min_value=1, max_value=180, value=65)
    theta_3dB = st.slider("Largeur de bande verticale de 3dB (degrés)", min_value=1, max_value=180, value=65)
    Am = st.slider("Rapport avant/arrière de l'antenne (dB)", min_value=1, max_value=50, value=30)
    SLA_v = st.slider("Niveau de rayonnement secondaire (dB)", min_value=1, max_value=50, value=30)
    G_Emax = st.slider("Gain maximal de l'élément unique (dB)", min_value=1, max_value=50, value=15)
    offset = st.slider("Décalage du gain (dB)", min_value=-50, max_value=50, value=0)
    # Convertir les angles en radians pour les calculs
    phi_3dB_rad = np.deg2rad(phi_3dB)
    theta_3dB_rad = np.deg2rad(theta_3dB)

    # Génération des données pour les angles
    phi_values = np.linspace(-180, 180, 360)
    theta_values = np.linspace(0, 180, 180)

    # Calculer le diagramme d'antenne horizontal et vertical
    A_EH_values = A_EH(phi_values, phi_3dB_rad, Am,offset)
    A_EV_values = A_EV(theta_values, theta_3dB_rad, SLA_v,offset)

    # Génération des données pour le diagramme 3D
    phi_3d = np.linspace(-180, 180, 360)
    theta_3d = np.linspace(0, 180, 180)
    phi_3d, theta_3d = np.meshgrid(phi_3d, theta_3d)
    A_E_values = A_E(phi_3d, theta_3d, G_Emax, phi_3dB_rad, theta_3dB_rad, Am, SLA_v,offset)
    if st.button("Diagramme 2D"):
        
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
        
       
    if st.button("Diagramme 3D"):
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
        # Ajout de la flèche de direction
        u, v, w = 0, 0, np.max(A_E_values)  # Direction maximale de rayonnement
        ax_3d.quiver(0, 0, 0, u, v, w, color='r', arrow_length_ratio=0.1)

        ax_3d.set_title("Diagramme 3D de l'antenne")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")

        st.pyplot(fig_3d)


with tab2:
    # Saisie des paramètres par l'utilisateur
    N_v = st.slider("Nombre de colonnes", min_value=1, max_value=10, value=4)
    N_h = st.slider("Nombre de lignes", min_value=1, max_value=10, value=4)
    d_v = st.slider("Espacement vertical (longueur d'onde)", min_value=0.1, max_value=2.0, value=0.5)
    d_h = st.slider("Espacement horizontal (longueur d'onde)", min_value=0.1, max_value=2.0, value=0.5)
    lam = st.slider("Longueur d'onde (m)", min_value=0.01, max_value=1.0, value=0.1)
    theta_i_max = st.slider("Angle d'inclinaison max (degrés)", min_value=0, max_value=180, value=65)
    phi_i_max = st.slider("Angle d'azimut max (degrés)", min_value=0, max_value=360, value=65)
    A_e = st.slider("Gain élémentaire (dB)", min_value=1, max_value=50, value=10)
    
    # Convertir les angles en radians pour les calculs
    theta_i_max_rad = np.deg2rad(theta_i_max)
    phi_i_max_rad = np.deg2rad(phi_i_max)

    # Génération des données pour les angles
    
    #theta_values = np.linspace(0, np.pi, 180)
    #phi_values = np.linspace(0, 2 * np.pi, 360)
    theta_values = np.linspace(0, 180, 180)
    phi_values = np.linspace(-180,180, 360)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)
   
    

    # Calcul du diagramme Antenne composite
    if st.button("Diagramme Composite 2D"):
        # Calcul du diagramme composite
        A_Beam_values = np.zeros_like(phi_values)  # Initialisation du tableau résultant

        for j in range(len(phi_values)):
            A_Beam_values[j] = A_Beam(theta_i_max, np.deg2rad(phi_values[j]), 
                                    A_e, N_v, N_h, d_v, d_h, lam, theta_i_max_rad, phi_i_max_rad)
                
        print("Shape of A_Beam_values:", A_Beam_values.shape)
        print("Shape of theta_values:", theta_values.shape)
        print("Shape of phi_values:", phi_values.shape)
        #A_Beam_values = A_Beam_values.T
       
        print(A_Beam_values)
        # Trouver l'index du gain maximal
       
        
        
        fig_r = plt.figure()
        ax_r = fig_r.add_subplot(111, projection='polar')
        phi_values_rad = np.deg2rad(phi_values)  # Conversion des degrés en radians pour le plot radial
        # Ajuster l'orientation de l'axe polaire pour que 0° soit en haut
        
        #for i in range(len(theta_values)):
        ax_r.plot(phi_values_rad,A_Beam_values)  # Utilisation des valeurs horizontales
        ax_r.set_title("Diagramme radial")
        #ax_r.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))  # Légende pour chaque courbe
        st.pyplot(fig_r)
        
    if st.button("Diagramme Composite 3D"):
        # Affichage du diagramme 3D
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Coordonnées sphériques vers cartésiennes
        X = A_Beam_values * np.sin(theta_values) * np.cos(phi_values)
        Y = A_Beam_values * np.sin(theta_values) * np.sin(phi_values)
        Z = A_Beam_values * np.cos(theta_values)

        ax_3d.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='viridis', edgecolor='none')

        # Ajout de la flèche de direction
        u, v, w = 0, 0, np.max(A_Beam_values)  # Direction maximale de rayonnement
        ax_3d.quiver(0, 0, 0, u, v, w, color='r', arrow_length_ratio=0.1)

        ax_3d.set_title("Diagramme 3D de l'antenne composite")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")

        st.pyplot(fig_3d)
        
    
