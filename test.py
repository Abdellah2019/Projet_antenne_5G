import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D


theta_values = np.linspace(0, np.pi, 180)
phi_values = np.linspace(0, 2 * np.pi, 360)
theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

print(theta_grid.shape)
print(phi_grid.shape)