#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[2]:


# medio exterior 0 (aire)
rho0 = 1.22 # kg/cm3
c0 = 343 # m/s
h0= 0.02
Z0 = rho0*c0/h0

def calcular_TMM(f, capas, Z0, c0, losses=True):
    """
    Calcula los coeficientes de reflexión y transmisión mediante TMM.
    
    Parámetros:
    - f: array de frecuencias (Hz)
    - capas: lista de tuplas (rho, c, h, L) para cada capa
    - Z0: impedancia del medio exterior
    - c0: velocidad del sonido en el medio exterior
    - losses: bool. Si True, incluye pérdidas viscotérmicas (modelo Stinson).
              Si False, modelo ideal sin pérdidas.
    
    Retorna:
    - R: coeficiente de reflexión (complejo)
    - T_ce: coeficiente de transmisión (con exponencial)
    - T_se: coeficiente de transmisión (sin exponencial)
    - T11, T22: elementos de la matriz de transferencia total
    """
    omega = 2*np.pi*f
    k0 = omega/c0
    
    T_tot = np.zeros((2, 2, np.size(f)), dtype=complex) # la matriz transferencia total la inicializo como la identidad 2x2, 
                                                      # y para TODAS las frecuencias del vector f (para calcular todo de golpe)
    T_tot[0, 0] = 1.0                                   
    T_tot[1, 1] = 1.0
    
    L_tot = 0
    
    # Constantes para el modelo de Stinson (solo se usan si losses=True)
    eta = 1.81*10**(-5)
    gamma = 1.4
    p0 = 101300 
    Pr = 0.7
    
    for (rho, c, h, L) in capas:
        if losses:
            # CON PÉRDIDAS (modelo de Stinson)
            Ge = np.sqrt(-1j*omega*rho/eta)
            Gk = np.sqrt(-1j*omega*rho*Pr/eta)
            rho_e = rho / (1 - (np.tanh(h/2 * Ge)) / (h/2 * Ge))
            kappa_e = gamma*p0 / (1 + (gamma - 1) * (np.tanh(h/2 * Gk)) / (h/2 * Gk))
            c_e = np.sqrt(kappa_e/rho_e)
            Z = rho_e*c_e/h
            k = omega/c_e
        else:
            # SIN PÉRDIDAS (modelo ideal)
            c_e = c  # velocidad del sonido real (sin pérdidas)
            Z = rho*c/h  # impedancia real
            k = omega/c_e  # número de onda real
        
        
        L_tot += L
        
        # matriz individual de la capa actual:
        M11 = np.cos(k*L)
        M12 = -1j*Z*np.sin(k*L)
        M21 = (-1j/Z)*np.sin(k*L)
        M22 = np.cos(k*L)
        
        # multiplicación de matrices: T_tot = M_actual * T_tot
        T_new = np.zeros_like(T_tot)
        
        T_new[0,0] = M11*T_tot[0,0] + M12*T_tot[1,0]
        T_new[0,1] = M11*T_tot[0,1] + M12*T_tot[1,1]
        T_new[1,0] = M21*T_tot[0,0] + M22*T_tot[1,0]
        T_new[1,1] = M21*T_tot[0,1] + M22*T_tot[1,1]
        T_tot = T_new
    
    T11, T12 = T_tot[0,0], T_tot[0,1]
    T21, T22 = T_tot[1,0], T_tot[1,1]
    
    denominador = T11 + T12/Z0 + T21*Z0 + T22
    R = (T11 + T12/Z0 - T21*Z0 - T22) /denominador
    T_ce = (2*np.exp(-1j*k0*L_tot)) /denominador
    T_se = 2/denominador # quitamos la exponencial, igual que en matriz de scattering
    
    return R, T_ce, T_se, T11, T22


# In[ ]:





# ## CASO 1: cavidad Fabry-Perot

# In[11]:


# capas interiores: (rho, c, h, L)   en este caso todas son aire
capas1 = [
    (1.22, 343, 0.005, 0.06),  # capa 1
]

f_min1 = 0
f_max1 = 12000
Nf1 = 200000

f1 = np.linspace(f_min1, f_max1, Nf1)

#R1, T_ce1, T_se1, T11_1, T22_1 = calcular_TMM(f1, capas1, Z0, c0)


# In[12]:


# Para el caso CON pérdidas 
R_loss1, T_ce_loss1, T_se_loss1, T11_loss1, T22_loss1 = calcular_TMM(f1, capas1, Z0, c0, losses=True)

# Para el caso SIN pérdidas (ideal)
R_ideal1, T_ce_ideal1, T_se_ideal1, T11_ideal1, T22_ideal1 = calcular_TMM(f1, capas1, Z0, c0, losses=False)


# In[15]:


# Módulos al cuadrado
Rcu_loss1 = np.abs(R_loss1)**2
Tcu_loss1 = np.abs(T_ce_loss1)**2
#Rcu_ideal1 = np.abs(R_ideal1)**2
#Tcu_ideal1 = np.abs(T_ce_ideal1)**2

# Absorción (solo para caso con pérdidas)
alpha1 = 1 - Rcu_loss1 - Tcu_loss1

# Figura (c): Caso sin pérdidas (ideal) 
plt.figure(figsize=(8, 6))
plt.plot(f1, np.abs(R_ideal1)**2, label='|R|² (sin pérdidas)')
plt.plot(f1, np.abs(T_ce_ideal1)**2, label='|T|² (sin pérdidas)')
plt.xlabel('f (Hz)')
plt.ylabel('Energía')
plt.title('(c) Conservación de la energía - modelo ideal')
plt.legend()
plt.grid(True)
plt.show()


# Figura (d): Caso con pérdidas + absorción 
plt.figure(figsize=(8, 6))
plt.plot(f1, np.abs(R_loss1), label='|R| (con pérdidas)')
plt.plot(f1, np.abs(T_ce_loss1), label='|T| (con pérdidas)')
plt.plot(f1, alpha1, 'g-', label='α = 1 − |R|² − |T|²')
plt.xlabel('f (Hz)')
plt.ylabel('Coeficientes')
plt.title('(d) Coeficientes con pérdidas y absorción')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# GRÁFICA 1: PARTES REALES
axes[0,0].plot(f1, np.real(R_loss1), label='Re(R)')
axes[0,0].plot(f1, np.real(T_ce_loss1), label='Re(T)')
axes[0,0].set_xlabel('f (Hz)')
axes[0,0].set_ylabel('Parte real')
axes[0,0].set_title('(a) Partes reales')
axes[0,0].legend()
axes[0,0].grid(True)

# GRÁFICA 2: PARTES IMAGINARIAS
axes[0,1].plot(f1, np.imag(R_loss1), label='Im(R)')
axes[0,1].plot(f1, np.imag(T_ce_loss1), label='Im(T)')
axes[0,1].set_xlabel('f (Hz)')
axes[0,1].set_ylabel('Parte imaginaria')
axes[0,1].set_title('(b) Partes imaginarias')
axes[0,1].legend()
axes[0,1].grid(True)

# GRÁFICA 3: MÓDULOS AL CUADRADO
axes[1,0].plot(f1, np.abs(R_ideal1)**2, label='|R|² (sin pérdidas)')
axes[1,0].plot(f1, np.abs(T_ce_ideal1)**2, label='|T|² (sin pérdidas)')
axes[1,0].set_xlabel('f (Hz)')
axes[1,0].set_ylabel('Energía')
axes[1,0].set_title('(c) Conservación de la energía - modelo ideal')
axes[1,0].legend()
axes[1,0].grid(True)


# Absorción (solo para caso con pérdidas)
alpha1 = 1 - Rcu_loss1 - Tcu_loss1


# GRÁFICA 4: VALORES ABSOLUTOS
axes[1,1].plot(f1, np.abs(R_loss1), label='|R| (con pérdidas)')
axes[1,1].plot(f1, np.abs(T_ce_loss1), label='|T| (con pérdidas)')
axes[1,1].plot(f1, alpha1, 'g-', label='α = 1 − |R|² − |T|²')
axes[1,1].set_xlabel('f (Hz)')
axes[1,1].set_ylabel('Coeficientes')
axes[1,1].set_title('(d) Coeficientes con pérdidas y absorción')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()

plt.savefig('mancuerna_teoricos.png')

plt.show()


# #### Comparación con los resultados experimentales:

# In[18]:


# Cargamos el archivo y aplanamos a 1D para asegurar que es un "churro" de datos
raw_data = np.loadtxt('FP_FP.txt').flatten()

# Calculamos cuántas frecuencias hay (dividimos entre 13 variables)
num_variables = 13
num_puntos = len(raw_data) // num_variables

# matlab suele guardar todos los valores de la primera variable, luego todos los de la segunda...
# Así que reshape(13, -1) crea una matriz de 13 filas y muchas columnas.
# .T (transponer) la gira para tener N filas (frecuencias) y 13 columnas.
datos = raw_data.reshape((num_variables, -1)).T

datos_reducidos = datos[::10, :] # para plotear una de cada 10 frecs experimentales

# ahora data ya es 2D y podemos acceder con [:, 0]
freq = datos_reducidos[:, 0]
real_R = datos_reducidos[:, 1]
imag_R = datos_reducidos[:, 2]
real_T = datos_reducidos[:, 3]
imag_T = datos_reducidos[:, 4]

# Cálculo de módulos (magnitud) experimentales
mod_R_exp = np.sqrt(real_R**2 + imag_R**2)
mod_T_exp = np.sqrt(real_T**2 + imag_T**2)


# absorción experimental alpha = 1 - |R|² - |T|²
alpha_exp = 1 - (mod_R_exp**2 + mod_T_exp**2)

plt.figure(figsize=(10, 6))
plt.plot(freq, mod_R_exp, label='|R|', color='tab:blue')
plt.plot(freq, mod_T_exp, label='|T|', color='tab:orange')

plt.title('Medidas experimentales - Guía 1', fontsize=14)
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Magnitud', fontsize=12)
plt.xlim([0, np.max(freq)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)

plt.show()


# In[20]:


# Módulos teóricos
mod_R_teo = np.abs(R_loss1)
mod_T_teo = np.abs(T_ce_loss1)

# calcular absorción teórica
alpha_teo = 1 - (mod_R_teo**2 + mod_T_teo**2)



plt.figure(figsize=(10, 6))  
# Curvas teóricas (líneas continuas)
plt.plot(f1, mod_R_teo, label='|R| teórico', color='tab:blue', linewidth=3)
plt.plot(f1, mod_T_teo, label='|T| teórico', color='tab:orange', linewidth=3)
plt.plot(f1, alpha_teo, label='α teórica', color='tab:green', linewidth=3)
# Curvas experimentales (puntos o líneas discontinuas)
plt.plot(freq, mod_R_exp, label='|R| experimental', linestyle=':', color='tab:blue', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:blue')
plt.plot(freq, mod_T_exp, label='|T| experimental', linestyle=':', color='tab:orange', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:orange')
plt.plot(freq, alpha_exp, label='α experimental', linestyle=':', color='tab:green', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:green')

plt.title('Comparación de los resultados teóricos y experimentales - Guía 1')
plt.xlabel('f (Hz)', fontsize=12)
plt.ylabel('Amplitud / Absorción', fontsize=12)
plt.xlim([0, np.max(freq)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.savefig('mancuerna_experim.png')
plt.show()


# In[ ]:





# In[ ]:





# ## CASO 2: cavidad resonante

# In[21]:


L1 = 0.04
L = 0.06

# capas interiores: (rho, c, h, L)   en este caso todas son aire
capas2 = [
    (1.22, 343, 0.005, (L-L1)/2),  # capa 1
    (1.22, 343, 0.02, L1),  # capa 2
    (1.22, 343, 0.005, (L-L1)/2)  # capa 3
]


f_min2 = 0
f_max2 = 12000
Nf2 = 200000

f2 = np.linspace(f_min2, f_max2, Nf2)

#R2, T_ce2, T_se2, T11_2, T22_2 = calcular_TMM(f2, capas2, Z0, c0)


# In[22]:


# Para el caso CON pérdidas
R_loss2, T_ce_loss2, T_se_loss2, T11_loss2, T22_loss2 = calcular_TMM(f2, capas2, Z0, c0, losses=True)

# Para el caso SIN pérdidas (ideal)
R_ideal2, T_ce_ideal2, T_se_ideal2, T11_ideal2, T22_ideal2 = calcular_TMM(f2, capas2, Z0, c0, losses=False)


# In[23]:


# Módulos al cuadrado
Rcu_loss2 = np.abs(R_loss2)**2
Tcu_loss2 = np.abs(T_ce_loss2)**2
#Rcu_ideal2 = np.abs(R_ideal2)**2
#Tcu_ideal2 = np.abs(T_ce_ideal2)**2

# Absorción (solo para caso con pérdidas)
alpha2 = 1 - Rcu_loss2 - Tcu_loss2

# figura (c): caso sin pérdidas (ideal) 
plt.figure(figsize=(8, 6))
plt.plot(f1, np.abs(R_ideal2)**2, label='|R|² (sin pérdidas)')
plt.plot(f1, np.abs(T_ce_ideal2)**2, label='|T|² (sin pérdidas)')
plt.xlabel('f (Hz)')
plt.ylabel('Energía')
plt.title('(c) Conservación de la energía - modelo ideal')
plt.legend()
plt.grid(True)
plt.show()


# fig (d): Caso con pérdidas + absorción 
plt.figure(figsize=(8, 6))
plt.plot(f1, np.abs(R_loss2), label='|R| (con pérdidas)')
plt.plot(f1, np.abs(T_ce_loss2), label='|T| (con pérdidas)')
plt.plot(f1, alpha2, 'g-', label='α = 1 − |R|² − |T|²')
plt.xlabel('f (Hz)')
plt.ylabel('Coeficientes')
plt.title('(d) Coeficientes con pérdidas y absorción')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[24]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# GRÁFICA 1: PARTES REALES
axes[0,0].plot(f2, np.real(R_loss2), label='Re(R)')
axes[0,0].plot(f2, np.real(T_ce_loss2), label='Re(T)')
axes[0,0].set_xlabel('f (Hz)')
axes[0,0].set_ylabel('Parte real')
axes[0,0].set_title('(a) Partes reales')
axes[0,0].legend()
axes[0,0].grid(True)

# GRÁFICA 2: PARTES IMAGINARIAS
axes[0,1].plot(f2, np.imag(R_loss2), label='Im(R)')
axes[0,1].plot(f2, np.imag(T_ce_loss2), label='Im(T)')
axes[0,1].set_xlabel('f (Hz)')
axes[0,1].set_ylabel('Parte imaginaria')
axes[0,1].set_title('(b) Partes imaginarias')
axes[0,1].legend()
axes[0,1].grid(True)

# GRÁFICA 3: MÓDULOS AL CUADRADO
axes[1,0].plot(f2, np.abs(R_ideal2)**2, label='|R|² (sin pérdidas)')
axes[1,0].plot(f2, np.abs(T_ce_ideal2)**2, label='|T|² (sin pérdidas)')
axes[1,0].set_xlabel('f (Hz)')
axes[1,0].set_ylabel('Energía')
axes[1,0].set_title('(c) Conservación de la energía - modelo ideal')
axes[1,0].legend()
axes[1,0].grid(True)

# Absorción (solo para el caso con pérdidas)
alpha2 = 1 - Rcu_loss2 - Tcu_loss2

# GRÁFICA 4: VALORES ABSOLUTOS
axes[1,1].plot(f2, np.abs(R_loss2), label='|R| (con pérdidas)')
axes[1,1].plot(f2, np.abs(T_ce_loss2), label='|T| (con pérdidas)')
axes[1,1].plot(f2, alpha2, 'g-', label='α = 1 − |R|² − |T|²')
axes[1,1].set_xlabel('f (Hz)')
axes[1,1].set_ylabel('Coeficientes')
axes[1,1].set_title('(d) Coeficientes con pérdidas y absorción')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()

plt.savefig('resonante_teoricos.png')

plt.show()


# In[ ]:





# #### Comparación con los datos experimentales:

# In[29]:


# Cargamos el archivo y aplanamos a 1D para asegurar que es un "churro" de datos
raw_data2 = np.loadtxt('FP_resonante.txt').flatten()

# Calculamos cuántas frecuencias hay (dividimos entre 13 variables)
num_variables2 = 13
num_puntos2 = len(raw_data2) // num_variables2


# matlab suele guardar todos los valores de la primera variable, luego todos los de la segunda...
# así que reshape(13, -1) crea una matriz de 13 filas y muchas columnas.
# .T (transponer) la gira para tener N filas (frecuencias) y 13 columnas.
datos2 = raw_data2.reshape((num_variables2, -1)).T

datos_reducidos2 = datos2[::10, :] # para plotear una de cada 10 frecs experimentales

# Ahora data ya es 2D y podemos acceder con [:, 0]
freq2 = datos_reducidos2[:, 0]
real_R2 = datos_reducidos2[:, 1]
imag_R2 = datos_reducidos2[:, 2]
real_T2 = datos_reducidos2[:, 3]
imag_T2 = datos_reducidos2[:, 4]

# Cálculo de módulos (magnitud)
mod_R2_exp = np.sqrt(real_R2**2 + imag_R2**2)
mod_T2_exp = np.sqrt(real_T2**2 + imag_T2**2)

# absorción experimental alpha = 1 - |R|² - |T|²
alpha_exp2 = 1 - (mod_R2_exp**2 + mod_T2_exp**2)


plt.figure(figsize=(10, 6))
plt.plot(freq2, mod_R2_exp, label='|R|', color='tab:blue')
plt.plot(freq2, mod_T2_exp, label='|T|', color='tab:orange')

plt.title('Medidas experimentales - Guía 2', fontsize=14)
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Magnitud', fontsize=12)
plt.xlim([0, np.max(freq2)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


# Módulos teóricos
mod_R_teo2 = np.abs(R_loss2)
mod_T_teo2 = np.abs(T_ce_loss2)

# Calcular absorción teórica
alpha_teo2 = 1 - (mod_R_teo2**2 + mod_T_teo2**2)



plt.figure(figsize=(10, 6)) 
# curvas teóricas (líneas continuas)
plt.plot(f2, mod_R_teo2, label='|R| teórico', color='tab:blue', linewidth=3)
plt.plot(f2, mod_T_teo2, label='|T| teórico', color='tab:orange', linewidth=3)
plt.plot(f2, alpha_teo2, label='α teórica', color='tab:green', linewidth=3)
# curvas experimentales (puntos o líneas discontinuas)
plt.plot(freq2, mod_R2_exp, label='|R| experimental', linestyle=':', color='tab:blue', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:blue')
plt.plot(freq2, mod_T2_exp, label='|T| experimental', linestyle=':', color='tab:orange', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:orange')
plt.plot(freq2, alpha_exp2, label='α experimental', linestyle=':', color='tab:orange', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:green')


plt.title('Comparación de los resultados teóricos y experimentales - Guía 2')
plt.xlabel('f (Hz)', fontsize=12)
plt.ylabel('Amplitud / Absorción', fontsize=12)
plt.xlim([0, np.max(freq2)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.savefig('resonante_experim.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## CASO 3: guía cristal fonónico

# In[31]:


# capas interiores: (rho, c, h, L)   en este caso todas son aire
capas3 = [
    (1.22, 343, 0.005, 0.005),  # capa 1
    (1.22, 343, 0.02, 0.01),  # capa 2
    (1.22, 343, 0.005, 0.01),  # capa 3
    (1.22, 343, 0.02, 0.01),  # capa 4
    (1.22, 343, 0.005, 0.01),  # capa 5
    (1.22, 343, 0.02, 0.01),  
    (1.22, 343, 0.005, 0.01),  # capa 7
    (1.22, 343, 0.02, 0.01),
    (1.22, 343, 0.005, 0.005),  # capa 9
]

capas4 =  [
    (1.22, 343, 0.02, 0.01),  # capa 1
    (1.22, 343, 0.005, 0.01) ]

f_min3 = 0
f_max3 = 12000
Nf3 = 200000

f3 = np.linspace(f_min3, f_max3, Nf3)

# R3, T_ce3, T_se3, t11, t22 = calcular_TMM(f3, capas3, Z0, c0)
R4, T_ce4, T_se4, T11_4, T22_4 = calcular_TMM(f3, capas4, Z0, c0)

# Para el caso CON pérdidas 
R_loss3, T_ce_loss3, T_se_loss3, T11_loss3, T22_loss3 = calcular_TMM(f3, capas3, Z0, c0, losses=True)

# Para el caso SIN pérdidas (ideal)
R_ideal3, T_ce_ideal3, T_se_ideal3, T11_ideal3, T22_ideal3 = calcular_TMM(f3, capas3, Z0, c0, losses=False)



L = sum(capa[-1] for capa in capas4)

K= (np.arccos((T11_4+T22_4)/2)) / L 
#KL= ((T11_4+T22_4)/2) 


# In[32]:


# Módulos al cuadrado
Rcu_loss3 = np.abs(R_loss3)**2
Tcu_loss3 = np.abs(T_ce_loss3)**2
#Rcu_ideal3 = np.abs(R_ideal3)**2
#Tcu_ideal3 = np.abs(T_ce_ideal3)**2

# Absorción (solo para caso con pérdidas)
alpha3 = 1 - Rcu_loss3 - Tcu_loss3

# fig (c): Caso sin pérdidas (ideal) 
plt.figure(figsize=(8, 6))
plt.plot(f3, np.abs(R_ideal3)**2, label='|R|² (sin pérdidas)')
plt.plot(f3, np.abs(T_ce_ideal3)**2, label='|T|² (sin pérdidas)')
plt.xlabel('f (Hz)')
plt.ylabel('Energía')
plt.title('(c) Conservación de la energía - modelo ideal')
plt.legend()
plt.grid(True)
plt.show()


# fig (d): Caso con pérdidas + absorción 
plt.figure(figsize=(8, 6))
plt.plot(f3, np.abs(R_loss3), label='|R| (con pérdidas)')
plt.plot(f3, np.abs(T_ce_loss3), label='|T| (con pérdidas)')
plt.plot(f3, alpha3, 'g-', label='α = 1 − |R|² − |T|²')
plt.xlabel('f (Hz)')
plt.ylabel('Coeficientes')
plt.title('(d) Coeficientes con pérdidas y absorción')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[33]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# GRÁFICA 1: PARTES REALES
axes[0,0].plot(f3, np.real(R_loss3), label='Re(R)')
axes[0,0].plot(f3, np.real(T_ce_loss3), label='Re(T)')
axes[0,0].set_xlabel('f (Hz)')
axes[0,0].set_ylabel('Parte real')
axes[0,0].set_title('(a) Partes reales')
axes[0,0].legend()
axes[0,0].grid(True)

# GRÁFICA 2: PARTES IMAGINARIAS
axes[0,1].plot(f3, np.imag(R_loss3), label='Im(R)')
axes[0,1].plot(f3, np.imag(T_ce_loss3), label='Im(T)')
axes[0,1].set_xlabel('f (Hz)')
axes[0,1].set_ylabel('Parte imaginaria')
axes[0,1].set_title('(b) Partes imaginarias')
axes[0,1].legend()
axes[0,1].grid(True)

# GRÁFICA 3: MÓDULOS AL CUADRADO
axes[1,0].plot(f3, np.abs(R_ideal3)**2, label='|R|² (sin pérdidas)')
axes[1,0].plot(f3, np.abs(T_ce_ideal3)**2, label='|T|² (sin pérdidas)')
axes[1,0].set_xlabel('f (Hz)')
axes[1,0].set_ylabel('Energía')
axes[1,0].set_title('(c) Conservación de la energía - modelo ideal')
axes[1,0].legend()
axes[1,0].grid(True)


# absorción (solo para caso con pérdidas)
alpha3 = 1 - Rcu_loss3 - Tcu_loss3

# GRÁFICA 4: VALORES ABSOLUTOS
axes[1,1].plot(f3, np.abs(R_loss3), label='|R| (con pérdidas)')
axes[1,1].plot(f3, np.abs(T_ce_loss3), label='|T| (con pérdidas)')
axes[1,1].plot(f3, alpha3, 'g-', label='α = 1 − |R|² − |T|²')
axes[1,1].set_xlabel('f (Hz)')
axes[1,1].set_ylabel('Coeficientes')
axes[1,1].set_title('(d) Coeficientes con pérdidas y absorción')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()

plt.savefig('cristal_teoricos.png')

plt.show()


# In[ ]:





# In[20]:


# GRÁFICA 5: RELACIÓN DE DISPERSIÓN
plt.figure(figsize=(8,6))
plt.plot(np.real(K), (f3), label='Re(K)')
plt.plot(np.abs(np.imag(K)), (f3), label='Im(K)')
plt.xlabel(r'$K (m^{-1})$')
plt.ylabel('f (Hz)')
plt.title('Relación de dispersión')
plt.legend()
plt.grid(True)
plt.savefig('cristal_bandgap.png')
plt.show()


# In[ ]:





# #### Comparación con los datos experimentales:

# In[35]:


# Cargamos el archivo y aplanamos a 1D para asegurar que es un "churro" de datos
raw_data3 = np.loadtxt('FP_Phononic.txt').flatten()

# Calculamos cuántas frecuencias hay (dividimos entre 13 variables)
num_variables3 = 13
num_puntos3 = len(raw_data3) // num_variables3


# matlab suele guardar todos los valores de la primera variable, luego todos los de la segunda...
# así que reshape(13, -1) crea una matriz de 13 filas y muchas columnas.
# .T (transponer) la gira para tener N filas (frecuencias) y 13 columnas.
datos3 = raw_data3.reshape((num_variables3, -1)).T

datos_reducidos3 = datos3[::10, :] # para plotear una de cada 10 frecs experimentales


# Ahora data ya es 2D y podemos acceder con [:, 0]
freq3 = datos_reducidos3[:, 0]
real_R3 = datos_reducidos3[:, 1]
imag_R3 = datos_reducidos3[:, 2]
real_T3 = datos_reducidos3[:, 3]
imag_T3 = datos_reducidos3[:, 4]

# Cálculo de módulos (magnitud)
mod_R3_exp = np.sqrt(real_R3**2 + imag_R3**2)
mod_T3_exp = np.sqrt(real_T3**2 + imag_T3**2)

# Calcular absorción experimental alpha = 1 - |R|² - |T|²
alpha_exp3 = 1 - (mod_R3_exp**2 + mod_T3_exp**2)



plt.figure(figsize=(10, 6))
plt.plot(freq3, mod_R3_exp, label='|R|', color='tab:blue')
plt.plot(freq3, mod_T3_exp, label='|T|', color='tab:orange')

plt.title('Medidas experimentales - Guía 3 (cristal fonónico)', fontsize=14)
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Magnitud', fontsize=12)
plt.xlim([0, np.max(freq3)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.show()


# In[36]:


# Módulos teóricos
mod_R_teo3 = np.abs(R_loss3)
mod_T_teo3 = np.abs(T_ce_loss3)

# absorción teórica
alpha_teo3 = 1 - (mod_R_teo3**2 + mod_T_teo3**2)


plt.figure(figsize=(10, 6)) 
# curvas teóricas (líneas continuas)
plt.plot(f3, mod_R_teo3, label='|R| teórico', color='tab:blue', linewidth=3)
plt.plot(f3, mod_T_teo3, label='|T| teórico', color='tab:orange', linewidth=3)
plt.plot(f3, alpha_teo3, label='α teórica', color='tab:green', linewidth=3)
# curvas experimentales (puntos o líneas discontinuas)
plt.plot(freq3, mod_R3_exp, label='|R| experimental', linestyle=':', color='tab:blue', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:blue')
plt.plot(freq3, mod_T3_exp, label='|T| experimental', linestyle=':', color='tab:orange', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:orange')
plt.plot(freq3, alpha_exp, label='α experimental', linestyle=':', color='tab:green', marker='o', 
         markersize=3, markerfacecolor='none', markeredgecolor='tab:green')

plt.title('Comparación de los resultados teóricos y experimentales - Guía 3 (cristal fonónico)')
plt.xlabel('f (Hz)', fontsize=12)
plt.ylabel('Amplitud / Absorción', fontsize=12)
plt.xlim([0, np.max(freq3)])
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.savefig('cristal_experim.png')
plt.show()


# In[ ]:




