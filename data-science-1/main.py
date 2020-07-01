#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[15]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[16]:


# Sua análise da parte 1 começa aqui.
df = dataframe.copy()
df.head()


# In[31]:


plt.title('Histograma das distribuições')
sns.distplot(df['normal'], color='g')
sns.distplot(df['binomial'], color='r')


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[20]:


def q1():
    # Retorne aqui o resultado da questão 1.
    q1_norm, q2_norm, q3_norm = tuple(df['normal'].quantile([.25, .5, .75]))
    q1_binom, q2_binom, q3_binom = tuple(df['binomial'].quantile([.25, .5, .75]))

    dif = (q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom)
    dif = tuple(map(lambda x: round(x, 3), dif))
    return dif


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?
# 
# > Em amostras suficientemente grandes, é possível aproximar as distribuições normal e binomial.

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[17]:


def q2():
    # Retorne aqui o resultado da questão 2.
    media, desv = df['normal'].mean(), df['normal'].std()
    ecdf = ECDF(df['normal'])
    prob = ecdf(media+desv)-ecdf(media-desv)
    res = round(prob, 3)

    return float(res)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.
# 
# > Sim, o valor é bastante próximo do teórico para a distribuição normal.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[54]:


def q3():
    # Retorne aqui o resultado da questão 3.
    m_norm, v_norm = df['normal'].mean(), df['normal'].var()
    m_binom, v_binom = df['binomial'].mean(), df['binomial'].var()

    dif = (m_binom - m_norm, v_binom - v_norm)
    res = tuple(map(lambda x: round(x, 3), dif)) # aplica round() nos elementos da tupla
    return res


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[2]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[5]:


# Sua análise da parte 2 começa aqui
stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[13]:


def q4():
    # Retorne aqui o resultado da questão 4.
    tmp = stars.loc[stars['target']==0,'mean_profile']
    false_pulsar_mean_profile_standardized = (tmp-tmp.mean())/tmp.std()

    quantis_teoricos = sct.norm.ppf([.8, .9, .95], loc=0, scale=1)
    ecdf = ECDF(false_pulsar_mean_profile_standardized)

    res = tuple(map(lambda x: round(x, 3), ecdf(quantis_teoricos)))
    return res


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    tmp = stars.loc[stars['target']==0,'mean_profile']
    false_pulsar_mean_profile_standardized = (tmp-tmp.mean())/tmp.std()

    quantis = [.25, .5, .75]
    cdf_q = sct.norm.ppf(quantis, loc=0, scale=1)
    ecdf_q = false_pulsar_mean_profile_standardized.quantile(quantis)

    dif = tuple(map(lambda i,j: i-j, ecdf_q, cdf_q))
    dif = tuple(map(lambda x: round(x, 3), dif))

    return dif


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
