#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.

# corrigindo textos
cols = ['Country','Region']
countries[cols] = countries[cols].apply(lambda col: col.str.strip())


# In[6]:


# convertendo numéricos
cols = ['Pop_density','Coastline_ratio', 'Net_migration', 'Infant_mortality','Literacy', 'Phones_per_1000', 'Arable','Climate','Crops', 'Other','Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

countries[cols] = countries[cols].apply(lambda col: col.str.replace(',', '.'))
countries[cols] = countries[cols].astype('float')


# In[7]:


countries.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[118]:


def q1():
    # Retorne aqui o resultado da questão 1.
    regions = countries['Region'].sort_values().unique().tolist()
    return regions


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


from sklearn.preprocessing import KBinsDiscretizer


# In[117]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discr = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    out = discr.fit_transform(countries['Pop_density'].values.reshape(-1, 1))
    count = np.sum(out==9.)
    return int(count)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[12]:


from sklearn.preprocessing import OneHotEncoder


# In[116]:


def q3():
    # Retorne aqui o resultado da questão 3.
    enc = OneHotEncoder(sparse=False)
    fit = enc.fit_transform(countries[['Region','Climate']].dropna())
    feats = fit.shape[1]
    feats += 1 # compensando uma categoria perdida com .dropna()

    return int(feats)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[123]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[124]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline


# In[129]:


def q4():
    # Retorne aqui o resultado da questão 4.
    
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('st',  StandardScaler())
    ])

    transf = ColumnTransformer([
        ('nu', FunctionTransformer(None), make_column_selector(dtype_exclude=['int','float'])),
        ('tr', pipe, make_column_selector(dtype_include=['int','float']))
    ])

    # aplica ao dataset completo
    trans_fit = transf.fit(countries)

    data = pd.DataFrame([test_country], columns=countries.columns)
    out = pd.DataFrame(transf.transform(data), columns=countries.columns)

    out_format = round(float(out['Arable']), 3)
    return out_format
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[111]:


countries['Net_migration'].describe()


# In[112]:


countries['Net_migration'].skew()


# In[109]:


sns.boxplot(y = countries['Net_migration'])


# De maneira geral, muitas observações fora do _interquartile range_.

# In[119]:


def q5():
    # Retorne aqui o resultado da questão 4.
    x = countries['Net_migration']
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1

    out_abaixo = x.between(-np.inf, q1-1.5*iqr).sum()
    out_acima = x.between(q3+1.5*iqr, np.inf).sum()

    return (int(out_abaixo), int(out_acima), False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[49]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[113]:


def q6():
    # Retorne aqui o resultado da questão 4.
    cvec = CountVectorizer()
    cvec_fit = cvec.fit_transform(newsgroup.data)

    # onde está 'phone'?
    index_phone = cvec.get_feature_names().index('phone')

    # matriz
    mat = cvec_fit.todense()
    res = mat[:, index_phone].sum()

    return int(res)


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[76]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[114]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tfidf = TfidfVectorizer()
    tfidf_fit = tfidf.fit_transform(newsgroup.data)

    # onde está 'phone'?
    index_phone = tfidf.get_feature_names().index('phone')

    # matriz
    mat = tfidf_fit.todense()
    res = mat[:, index_phone].sum()

    return round(float(res), 3)

