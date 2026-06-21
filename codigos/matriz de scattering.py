#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

c1 = 343  
c2 = 1500   
L = 0.1     

# CREACIÓN DE LA MALLA COMPLEJA (El plano fR + i fI)

# creamos vectores para el eje X (real) y el eje Y (imaginario)
fR_vals = np.linspace(10, 40000, 400)   # parte real de 10 Hz a 40 kHz
fI_vals = np.linspace(-2000, 2000, 400) # parte imaginaria de -2 kHz a +2 kHz

f_R, f_I = np.meshgrid(fR_vals, fI_vals) # np.meshgrid cruza ambos vectores para crear una cuadrícula de 400x400 puntos

# construimos nuestra frecuencia compleja
f_complex = f_R + 1j*f_I
omega = 2*np.pi*f_complex


# CÁLCULO DE LA MATRIZ DE SCATTERING

# Los números de onda ahora también serán matrices de 400x400 puntos complejos
k1 = omega/c1
k2 = omega/c2

# las fórmulas de siempre
denom = 2*k1*k2*np.cos(L*k2) - 1j*np.sin(L*k2)*(k1**2 + k2**2)
numer_R = -1j*np.sin(L*k2)*(k1**2 - k2**2)
#numer_T = 2*k1*k2*np.exp(-1j*L*k1)
numer_T = 2*k1*k2
R = numer_R/denom
T = numer_T/denom

# Calculamos los autovalores de la matriz S
lambda_mas = T+R
lambda_menos = T-R


# VISUALIZACIÓN EN MAPAS DE CALOR
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
# parámetros visuales comunes
cmap_choice = 'magma'

# Gráfica 1: Autovalor lambda_+ 
ax1 = axes[0]
im1 = ax1.pcolormesh(f_R, f_I, np.abs(lambda_mas), shading='auto', cmap=cmap_choice, norm=LogNorm(vmin=0.01, vmax=100))
                                                       # Usamos LogNorm para que los polos (infinitos) no rompan la gráfica
ax1.set_title(r'Módulo del Autovalor $|\lambda_+| = |T + R|$')
ax1.set_ylabel('Frecuencia Imaginaria $f_I$ (Hz)')
ax1.axhline(0, color='white', linestyle='--', alpha=0.5) 
fig.colorbar(im1, ax=ax1, label='Amplitud (Escala Log)')

# Gráfica 2: Autovalor lambda_- 
ax2 = axes[1]
im2 = ax2.pcolormesh(f_R, f_I, np.abs(lambda_menos), shading='auto', cmap=cmap_choice, norm=LogNorm(vmin=0.01, vmax=100))
ax2.set_title(r'Módulo del Autovalor $|\lambda_-| = |T - R|$')
ax2.set_xlabel('Frecuencia Real $f_R$ (Hz)')
ax2.set_ylabel('Frecuencia Imaginaria $f_I$ (Hz)')
ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
fig.colorbar(im2, ax=ax2, label='Amplitud (Escala Log)')
plt.savefig('frec_compleja.png')


# In[ ]:




