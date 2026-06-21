#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ## Relación de dispersión

# In[ ]:


c = 343 #velocidad sonido (m/s)
h = 0.1 #altura guía (m)
n_modos = 4  # de 0 a 3
f_max = 12000 
Nf = 500  # núm. de puntos en frecuencia

f = np.linspace(1, f_max, Nf)  #vector de frecuencias (Hz)
omega = 2 * np.pi * f          
k = omega / c  #núm onda total (rad/m)

# inicializar arrays para kx (real e imaginario)
kx_real = np.zeros((Nf, n_modos))
kx_imag = np.zeros((Nf, n_modos))

# calcular kx para cada modo n
for n in range(n_modos):
    ky = n * np.pi / h  # número de onda vertical
    kx2 = k**2 - ky**2  # kx^2 = k^2 - ky^2
    kx = np.sqrt(kx2 + 0j)  # el 0j es porque kx puede ser complejo
    kx_real[:, n] = np.real(kx)  
    kx_imag[:, n] = np.imag(kx) 
    
    
    
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 fila, 2 columnas 

# panel izquierdo: parte imaginaria (líneas discontinuas)  
ax = axes[0]
for n in range(n_modos):
    ax.plot(kx_imag[:, n], f, '--', label=f'n={n}') 

ax.invert_xaxis()    
ax.set_xlabel(r'Im($k_x$)')
ax.set_ylabel('Frecuencia (Hz)')
ax.set_title(r'Parte imaginaria de $k_x$ (modos evanescentes)')
ax.legend()
ax.grid(True)

# panel derecho: parte real (líneas continuas) 
ax = axes[1]
for n in range(n_modos):
    ax.plot(kx_real[:, n], f, '-', label=f'n={n}')
ax.set_xlabel(r'Re($k_x$)')
ax.set_ylabel('Frecuencia (Hz)')
ax.set_title(r'Parte real de $k_x$ (modos propagantes)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('relacion_dispersion.png')

plt.show()


# In[ ]:





# ## Formas modales

# In[17]:


frecuencias = [1000, 1700, 2500, 4000]  # Hz

Lx = 1.0  # longitud en x (m)
Nx = 300  # número de puntos en la dirección x
Ny = 200  # ídem en y
h = 0.1 #altura guía (m)
c = 343 #velocidad sonido (m/s)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, h, Ny)

X, Y = np.meshgrid(x, y)  # crear la malla 2D

modos = [0, 1, 2, 3]


for f_sel in frecuencias:  # porque quiero generar una figura por frecuencia 
    omega_sel = 2*np.pi*f_sel
    k_sel = omega_sel / c # núm onda total
    
    fig, axes = plt.subplots(len(modos), 1, figsize=(6, 10), sharex=True)  # dentro de cada figura: 4 filas y 1 columna
    
    for i, n in enumerate(modos):  # recorre los modos, siendo i el índice y n el modo
        ky = n * np.pi / h # núm onda vertical 
        kx2 = k_sel**2 - ky**2
        kx = np.sqrt(kx2 + 0j) # recordemos que el 0j se pone para permitir complejos 
        # campo acústico
        p = np.real(np.exp(1j*kx*X) * np.cos(n*np.pi*Y/h))  # en exp: si kx es R --> propagación, si es C --> decaimiento
        ax = axes[i]  # para seleccionar la fila 
        im = ax.pcolormesh(X, Y, p, shading='auto', cmap='RdBu')
        ax.set_ylabel('y (m)')
        ax.set_title(f'n={n}')
    
    axes[-1].set_xlabel('x (m)')  # para solo poner el título del eje X una vez (abajo del todo)
    
    fig.suptitle(f'f = {f_sel} Hz')
    
    plt.tight_layout()
    plt.savefig('formas_modales.png')
    plt.show()


# In[24]:


from matplotlib.gridspec import GridSpec


frecuencias = [1000, 1700, 2500, 4000]

fig = plt.figure(figsize=(12, 12))
gs = GridSpec(8, 2, figure=fig)  # 8 filas (4 modos × 2) y 2 columnas

for idx_f, f_sel in enumerate(frecuencias):

    omega_sel = 2*np.pi*f_sel
    k_sel = omega_sel / c

    # posición de la frecuencia en la cuadrícula
    bloque_fila = (idx_f // 2) * 4   # 0 o 4
    bloque_col = idx_f % 2           # 0 o 1

    for n in modos:

        ky = n*np.pi/h
        kx2 = k_sel**2 - ky**2
        kx = np.sqrt(kx2 + 0j)

        p = np.real(np.exp(1j*kx*X) * np.cos(n*np.pi*Y/h))

        ax = fig.add_subplot(gs[bloque_fila + n, bloque_col])

        im = ax.pcolormesh(X, Y, p, shading='auto', cmap='RdBu')

        ax.set_ylabel('y(m)')
        ax.set_title(f'n={n}')
        ax.set_xlabel(f'x(m)')

        if n == 0:
            ax.set_title(f'f = {f_sel} Hz\nn = {n}')
        else:
            ax.set_title(f'n = {n}')
            

        

plt.tight_layout()
plt.savefig('formas_modales.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Problemas 1 y 2

# In[2]:


rho1 = 1.225   # densidad aire (kg/m^3)  
c1 = 343     # velocidad sonido (m/s)
rho2 = 1000   # densidad agua (kg/m^3)  
c2 = 1500   # velocidad sonido  en agua (m/s)
L = 0.1   # espesor de la capa (m)
h1=0.03
h2=0.05

f_min = 1
f_max = 30000
Nf = 200000

f = np.linspace(f_min, f_max, Nf)
omega = 2*np.pi*f

k1 = omega/c1
k2 = omega/c2

Z1 = rho1*c1/h1
Z2 = rho2*c2/h2

# PROBLEMA 1
R1 = (Z2-Z1) / (Z2+Z1)
T1 = 2*Z2 / (Z2+Z1)
# convertimos en arrays del mismo tamaño que omega
R1 = R1 * np.ones_like(omega, dtype=complex) # R1 era un escalar, omega un array, matplotlib necesita 
T1 = T1 * np.ones_like(omega, dtype=complex) # arrays del mismo tamaño para plotear


# PROBLEMA 2

#denominador = 2*(k1*h1/rho1)*(k2*h2/rho2)*np.cos(L*k2) - 1j*np.sin(L*k2)*((k1*h1/rho1)**2+(k2*h2/rho2)**2)
#numerador_R = -1j*np.sin(L*k2)*((k1*h1/rho1)**2-(k2*h2/rho2)**2)
#R_trig = numerador_R / denominador
#numerador_T = 2*(k1*h1/rho1)*(k2*h2/rho2)*np.exp(-1j*L*k1)
#T_trig = numerador_T / denominador

denominador = 2*k1*k2*np.cos(L*k2) - 1j*np.sin(L*k2)*(k1**2+k2**2)
numerador_R = -1j*np.sin(L*k2)*(k1**2-k2**2)
R_trig = numerador_R / denominador
numerador_T = 2*k1*k2*np.exp(-1j*L*k1)
T_trig = numerador_T / denominador


R2_2 = R_trig
T2_2 = T_trig


# GRÁFICA 1: PARTES REALES
plt.figure(figsize=(8,6))
plt.plot(f, np.real(R2_2), label='Re(R)')
plt.plot(f, np.real(T2_2), label='Re(T)')
plt.xlabel('f (Hz)')
plt.ylabel('Parte real')
plt.title('Partes reales')
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICA 2: PARTES IMAGINARIAS
plt.figure(figsize=(8,6))
plt.plot(f, np.imag(R2_2), label='Im(R)')
plt.plot(f, np.imag(T2_2), label='Im(T)')
plt.xlabel('f (Hz)')
plt.ylabel('Parte imaginaria')
plt.title('Partes imaginarias')
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICA 3: ENERGÍA 
plt.figure(figsize=(8,6))
plt.plot(f, np.abs(R2_2)**2, label=r'$|R|^2$')
plt.plot(f, (np.abs(T2_2)**2), label=r'$|T|^2$')
plt.plot(f, np.abs(R2_2)**2 + np.abs(T2_2)**2, label='suma')
plt.xlabel('f (Hz)')
plt.ylabel('Energía')
plt.title('Módulos al cuadrado: conservación de la energía')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# GRÁFICA 1: PARTES REALES
axes[0].plot(f, np.real(R2_2), label='Re(R)')
axes[0].plot(f, np.real(T2_2), label='Re(T)')
axes[0].set_xlabel('f (Hz)')
axes[0].set_ylabel('Parte real')
axes[0].set_title('(a) Partes reales')
axes[0].legend()
axes[0].grid(True)

# GRÁFICA 2: PARTES IMAGINARIAS
axes[1].plot(f, np.imag(R2_2), label='Im(R)')
axes[1].plot(f, np.imag(T2_2), label='Im(T)')
axes[1].set_xlabel('f (Hz)')
axes[1].set_ylabel('Parte imaginaria')
axes[1].set_title('(b) Partes imaginarias')
axes[1].legend()
axes[1].grid(True)

# GRÁFICA 3: ENERGÍA
axes[2].plot(f, np.abs(R2_2)**2, label=r'$|R|^2$')
axes[2].plot(f, np.abs(T2_2)**2, label=r'$|T|^2$')
axes[2].plot(f, np.abs(R2_2)**2 + np.abs(T2_2)**2, label=r'$|T|^2 +|R|^2 = 1$')
axes[2].set_xlabel('f (Hz)')
axes[2].set_ylabel('Energía')
axes[2].set_title('(c) Conservación de la energía')
axes[2].legend(loc='lower left')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('RT_problema2.png')
plt.show()


# In[ ]:





# ## Representación de p(x)

# In[33]:


f0 = 5000  # Hz
omega0 = 2*np.pi*f0

k1_0 = omega0/c1
k2_0 = omega0/c2

# recalculamos R y T para esa frecuencia
den0 = 2*k1_0*k2_0*np.cos(L*k2_0) - 1j*np.sin(L*k2_0)*(k1_0**2+k2_0**2)
R0 = (-1j*np.sin(L*k2_0)*(k1_0**2-k2_0**2)) / den0
T0 = (2*k1_0*k2_0*np.exp(-1j*L*k1_0)) / den0

# y también C y D 
C_trig0 = k1_0*(k1_0+k2_0)*np.exp(-1j*k2_0*L) / den0
D_trig0 = k1_0*(k2_0-k1_0)*np.exp(1j*k2_0*L) / den0

x1 = np.linspace(-3*L, 0, 400) # ejes espaciales
x2 = np.linspace(0, L, 400)
x3 = np.linspace(L, L+3*L, 400)

# campos en cada región
p1 = np.exp(1j*k1_0*x1) + R0*np.exp(-1j*k1_0*x1)
p2 = C_trig0*np.exp(1j*k2_0*x2) + D_trig0*np.exp(-1j*k2_0*x2)
p3 = T0*np.exp(1j*k1_0*x3)


# gráfica 1 
plt.figure(figsize=(8,6))
plt.plot(x1, p1, label='p1(x)')
plt.plot(x2, p2, label='p2(x)')
plt.plot(x3, p3, label='p3(x)')

plt.axvline(0, color='k', linestyle='--')
plt.axvline(L, color='k', linestyle='--')

plt.title(f'p(x) para f = {f0} Hz')
plt.xlabel('x (m)')
plt.ylabel('p(x)')
plt.legend()
plt.grid(True)
plt.show()


# gráfica 2
plt.figure(figsize=(8,6))
plt.plot(x1, np.abs(p1), label='|p1(x)|')
plt.plot(x2, np.abs(p2), label='|p2(x)|')
plt.plot(x3, np.abs(p3), label='|p3(x)|')

plt.axvline(0, color='k', linestyle='--')
plt.axvline(L, color='k', linestyle='--')

plt.title(f'Módulo de p(x) para f = {f0} Hz')
plt.xlabel('x (m)')
plt.ylabel('|p(x)|')
plt.legend()
plt.grid(True)
plt.savefig('campo_presiones.png')
plt.show()


# In[35]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GRÁFICA 1: campo de presión
axes[0].plot(x1, np.real(p1), label=r'$p_1(x)$')
axes[0].plot(x2, np.real(p2), label=r'$p_2(x)$')
axes[0].plot(x3, np.real(p3), label=r'$p_3(x)$')

axes[0].axvline(0, color='k', linestyle='--')
axes[0].axvline(L, color='k', linestyle='--')

axes[0].set_title(f'(a) p(x) para f = {f0} Hz')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('p(x)')
axes[0].legend()
axes[0].grid(True)

# GRÁFICA 2: módulo
axes[1].plot(x1, np.abs(p1), label=r'$|p_1(x)|$')
axes[1].plot(x2, np.abs(p2), label=r'$|p_2(x)|$')
axes[1].plot(x3, np.abs(p3), label=r'$|p_3(x)|$')

axes[1].axvline(0, color='k', linestyle='--')
axes[1].axvline(L, color='k', linestyle='--')

axes[1].set_title(f'(b) Módulo de p(x) para f = {f0} Hz')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('|p(x)|')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('campo_presiones.png')
plt.show()


# In[ ]:





# In[ ]:





# ### Si solo quiero ver las resonancias:

# In[11]:


N_modos = 4 # núm de resonancias que queremos ver (las primeras)

for n in range(1, N_modos+1):
    f_res = n*c2/(2*L) # frec resonante
    omega_res = 2*np.pi*f_res

    k1_res = omega_res/c1
    k2_res = omega_res/c2

    # coeficientes R y T
    denom_res = 2*k1_res*k2_res*np.cos(L*k2_res) - 1j*np.sin(L*k2_res)*(k1_res**2+k2_res**2)

    R_res = -1j*np.sin(L*k2_res)*(k1_res**2-k2_res**2) / denom_res
    T_res = 2*k1_res*k2_res*np.exp(-1j*L*k1_res) / denom_res

    # coeficientes dentro de la capa/pared
    C_res = k1_res*(k1_res+k2_res)*np.exp(-1j*k2_res*L) / denom_res
    D_res = k1_res*(k2_res-k1_res)*np.exp(1j*k2_res*L) / denom_res

    # presión en cada región
    p1_res = np.exp(1j*k1_res*x1) + R_res*np.exp(-1j*k1_res*x1)
    p2_res = C_res*np.exp(1j*k2_res*x2) + D_res*np.exp(-1j*k2_res*x2)
    p3_res = T_res*np.exp(1j*k1_res*x3)
    
    # gráfica 1
    plt.figure(figsize=(9,5))

    plt.plot(x1, p1_res, label="p1_res(x)")
    plt.plot(x2, p2_res, label="p2_res(x)")
    plt.plot(x3, p3_res, label="p3_res(x)")

    plt.axvline(0, color='k', linestyle='--')
    plt.axvline(L, color='k', linestyle='--')

    plt.title(f"Modo resonante n = {n}   (f = {f_res} Hz)")
    plt.xlabel("x (m)")
    plt.ylabel("p(x)")

    plt.grid(True)
    plt.legend()
    plt.show()
    
    # gráfica 2
    plt.figure(figsize=(9,5))
    plt.plot(x1, np.abs(p1_res), label='|p1(x)_res|')
    plt.plot(x2, np.abs(p2_res), label='|p2(x)_res|')
    plt.plot(x3, np.abs(p3_res), label='|p3(x)_res|')

    plt.axvline(0, color='k', linestyle='--')
    plt.axvline(L, color='k', linestyle='--')

    plt.title(f'Módulo de p(x) resonante para f = {f_res} Hz')
    plt.xlabel('x (m)')
    plt.ylabel('|p(x)_res|')
    plt.legend()
    plt.grid(True)
    plt.show()


# #### tiene sentido que en el aire tengamos líneas rectas porque justo en resonancia sen()=0 --> R=0=B, por lo tanto no se refleja nada. Si no hay onda reflejada, no hay ondas estacionarias fuera de la pared; toda la energía fluye limpiamente, T=1

# In[ ]:





# In[8]:


# para ver las ondas solo dentro de la pared: 

# eje espacial solo dentro de la capa
x = np.linspace(0, L, 800)

num_modos = 4

plt.figure(figsize=(9,6))


for n in range(1, num_modos+1):
    f_res2 = n*c2/(2*L)
    omega_res2 = 2*np.pi*f_res2

    k1_res2 = omega_res2/c1
    k2_res2 = omega_res2/c2

    # coeficientes
    denom_res2 = 2*k1_res2*k2_res2*np.cos(L*k2_res2) - 1j*np.sin(L*k2_res2)*(k1_res2**2+k2_res2**2)

    R_res2 = -1j*np.sin(L*k2_res2)*(k1_res2**2-k2_res2**2) / denom_res2
    T_res2 = 2*k1_res2*k2_res2*np.exp(-1j*L*k1_res2) / denom_res2

    C_res2 = k1_res2*(k1_res2+k2_res2)*np.exp(-1j*k2_res2*L) / denom_res2
    D_res2 = k1_res2*(k2_res2-k1_res2)*np.exp(1j*k2_res2*L) / denom_res2

    # campo dentro de la capa
    p = C_res2*np.exp(1j*k2_res2*x) + D_res2*np.exp(-1j*k2_res2*x)

    # normalizar para comparar modos
    p = p/np.max(np.abs(p))

    plt.plot(x, p, label=f"n = {n}")
    
   
    

plt.axvline(0, color='k', linestyle='--')
plt.axvline(L, color='k', linestyle='--')

plt.xlabel("x (m)")
plt.title("Modos resonantes dentro de la capa")
plt.legend()
plt.grid(True)
plt.show()


 # gráfica 2
plt.figure(figsize=(9,5))
plt.plot(x, np.abs(p), label='|p(x)|')

plt.axvline(0, color='k', linestyle='--')
plt.axvline(L, color='k', linestyle='--')

plt.xlabel('x (m)')
plt.ylabel('|p(x)|')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




