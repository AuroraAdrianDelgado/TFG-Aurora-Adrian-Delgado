#!/usr/bin/env python
# coding: utf-8

# ## Resolución del sistema de ecuaciones

# In[2]:


import sympy as sp

B, C, D, E = sp.symbols('B C D E')
k1, k2, L = sp.symbols('k1 k2 L', real=True)
i = sp.I

# definición de las ecuaciones:
ec1 = sp.Eq(1 + B, C + D)
ec2 = sp.Eq(i*k1 - B*i*k1, C*i*k2 - D*i*k2)
ec3 = sp.Eq(C*sp.exp(i*k2*L) + D*sp.exp(-i*k2*L), E*sp.exp(i*k1*L))
ec4 = sp.Eq(C*i*k2*sp.exp(i*k2*L) - D*i*k2*sp.exp(-i*k2*L), E*i*k1*sp.exp(i*k1*L))

# resolvemos el sistema:
solucion = sp.solve((ec1, ec2, ec3, ec4), (B, C, D, E), simplify=True)

B_sol = sp.simplify(solucion[B])
E_sol = sp.simplify(solucion[E])
C_sol = sp.simplify(solucion[C])
D_sol = sp.simplify(solucion[D])

B_sol


# In[3]:


E_sol


# In[4]:


C_sol


# In[5]:


D_sol


# In[6]:


R = solucion[B]
T = solucion[E]

check = sp.simplify(sp.expand_complex(R*sp.conjugate(R) + T*sp.conjugate(T)))
sp.simplify(check)


# In[ ]:





# ### Quiero ponerlo en forma trigonométrica:

# In[7]:


denominador = 2*k1*k2*sp.cos(L*k2) - 1j*sp.sin(L*k2)*(k1**2+k2**2)
numerador_R = -1j*sp.sin(L*k2)*(k1**2-k2**2)
R_trig = numerador_R / denominador
numerador_T = 2*k1*k2*sp.exp(-1j*L*k1)
T_trig = numerador_T / denominador

R_trig


# In[8]:


T_trig


# In[9]:


check_trig = sp.simplify(sp.expand_complex(R_trig*sp.conjugate(R_trig) + T_trig*sp.conjugate(T_trig)))
sp.simplify(check_trig)


# #### Esto demuestra que las transformaciones a forma trigonométrica son correctas!!

# In[12]:


T_2=T_trig*sp.conjugate(T_trig)
T_2


# In[13]:


sp.simplify(T_2)


# In[ ]:





# ## CORRECCIONES

# In[6]:


import sympy as sp

B, C, D, E = sp.symbols('B C D E')
k1, k2, L, k1p, k2p = sp.symbols('k1 k2 L K1 K2', real=True)
i = sp.I

# definición de las ecuaciones:
ec1 = sp.Eq(1 + B, C + D)
ec2 = sp.Eq(k1p - B*k1p, C*k2p - D*k2p)
ec3 = sp.Eq(C*sp.exp(i*k2*L) + D*sp.exp(-i*k2*L), E*sp.exp(i*k1*L))
ec4 = sp.Eq(C*k2p*sp.exp(i*k2*L) - D*k2p*sp.exp(-i*k2*L), E*k1p*sp.exp(i*k1*L))

# resolvemos el sistema:
solucion = sp.solve((ec1, ec2, ec3, ec4), (B, C, D, E), simplify=True)

B_sol = sp.simplify(solucion[B])
E_sol = sp.simplify(solucion[E])
C_sol = sp.simplify(solucion[C])
D_sol = sp.simplify(solucion[D])

B_sol


# ###### siendo K_i = k_i * h_i / rho_i

# In[7]:


E_sol


# In[8]:


C_sol


# In[9]:


D_sol


# In[10]:


R = solucion[B]
T = solucion[E]

check = sp.simplify(sp.expand_complex(R*sp.conjugate(R) + T*sp.conjugate(T)))
sp.simplify(check)


# ### Forma trigonométrica:

# In[11]:


denominador = 2*k1p*k2p*sp.cos(L*k2) - 1j*sp.sin(L*k2)*(k1p**2+k2p**2)
numerador_R = -1j*sp.sin(L*k2)*(k1p**2-k2p**2)
R_trig = numerador_R / denominador
numerador_T = 2*k1p*k2p*sp.exp(-1j*L*k1)
T_trig = numerador_T / denominador

R_trig


# In[12]:


T_trig


# In[13]:


check_trig = sp.simplify(sp.expand_complex(R_trig*sp.conjugate(R_trig) + T_trig*sp.conjugate(T_trig)))
sp.simplify(check_trig)


# #### está bien!!!! :)

# In[ ]:




