#!/usr/bin/env python
# coding: utf-8

# # Quality compression (noisy signal compression). Proof of concept (POC) for applying the LSTM model to a quality array.
# 
# ## The Purpose of the Experiment
# The purpose of the experiment is approximate (in the sense of classification) the sequencing data (incl. sequence and quality aspects) stored in the FASTA format using Long-Short Term Memory Artificial Neural Network (LSTM NN).
# 
# ## Description
# We use the $N=500$ past observations of the one-hot-encoded Sequences and Quality data to predict the next element of Sequence and Quality (aspect). We use $m=4$ units LSTM NN with $\tau=25$ timesteps as a classifier.
# 
# ## Assumptions
# We approximate finite sequences (with $N=500$ observations). We hope the LSTM NN classifies the endless sequences, but the LSTM NN insight might be represented in a compact form using the LSTM architecture. (If these do not work, we consider error terms from classification may be defined in a compact form).
# 
# ## Limitations
# We use only the $N=500$ past observations of the data. No {Mutations, Insertions, inversions, etc.} flags are used in the current implementation.

# In[52]:


from Bio import SeqIO
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# NN NN Architectrue
from keras.models import Model#, Sequential
from keras.layers import Input, LSTM, Flatten, Dense, Permute#, Dropout


# In[30]:


def seq2bool(seq, amines):
    return (np.expand_dims(np.array(list(seq)),1) == np.expand_dims(amines,1).T)

def qual2bool(qual):
    uq = np.unique(qual)
    return (np.expand_dims(qual,1) == np.expand_dims(uq,1).T)
amines = ['A', 'C', 'G', 'T']


# In[31]:


get_ipython().system('head -n 12 ../data/E_4_20_1_short_example.fq')


# In[32]:


qual_array, sequence_array = [],[]
with open("../data/E_4_20_1_short_example.fq") as input_handle:
    for read in SeqIO.parse(input_handle, "fastq"):
        sequence_array.append(seq2bool(read.seq, amines))
        qual_array.extend(read._per_letter_annotations["phred_quality"])

# Concatenate sequence
sequence_array = 1*np.concatenate(sequence_array, axis = 0)


# In[33]:


quality = pd.DataFrame.from_dict(
    {
        "range": np.arange(len(qual_array)),
        "signal": np.array(qual_array),
    }
)

sequence = pd.DataFrame.from_dict(
    {
        "range": np.arange(len(sequence_array)),
        "signal": np.argmax(sequence_array, axis = 1),
    }
)


# In[34]:


fig = px.scatter(quality, x="range", y="signal", trendline="rolling", 
                 trendline_options=dict(window=100, win_type="gaussian", function_args=dict(std=2)), 
                 trendline_color_override="red"
)
fig.show()


# In[35]:


fig = px.area(sequence[0:1000]-2, x="range", y="signal")
fig.show()


# In[36]:


fig = px.histogram(quality, x="signal")
fig.show()


# In[37]:


fig = px.histogram(sequence, x="signal")
fig.show()


# In[38]:


fig = px.line(
    pd.DataFrame.from_dict(
        {
            "signal": qual_array[:1000],
        }
    ),
    y="signal"
)
fig.show()


# In[39]:


method = 'q+s'; #'s', 'q+s'
N = 5*10**2;

if method == 's':
    # Quality
    x = qual2bool(np.array([qual_array[:N]]).T)

elif method == 's':    
    # Sequence
    x = np.moveaxis(np.expand_dims(sequence_array[:N,:],2), 1,2)

elif method == 'q+s':
    # Sequence + Quality
    x1 = np.moveaxis(np.expand_dims(sequence_array[:N,:],2), 1,2)
    x2 = qual2bool(np.array([qual_array[:N]]).T)
    x = np.concatenate([x1,x2], axis = 2)

t = np.roll(x,1)


# In[40]:


# Vectorizations (one-hot encoding likewise)
timesteps = 25;
regressors = 5;

X = [];
for i in range(timesteps):
    X.append(np.roll(x,-i))

X = np.concatenate(X, axis = 1)
t = t[:,0:1,0:regressors];


t = np.moveaxis(t, 2, 1); #TO BE REMOVED


# In[41]:


# Architecture
units = 4;

i = Input(shape=(timesteps,X.shape[2]))
l = LSTM(units, return_sequences = True, use_bias = True) (i)
l *= (2.0)/np.tanh(1).astype(float);
l = Flatten()(l)
l = Dense(regressors, activation = 'relu') (l)

regressor = Model (inputs = i, outputs = l)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])
regressor.fit(X, t, epochs = 2*10**3, verbose = 0)


# In[42]:


regressor_df = pd.DataFrame.from_dict(
    {
        'loss': regressor.history.history['loss'],
    }
)
fig = px.line(regressor_df, y="loss")
fig.show()


# In[43]:


# Architecture
regressor.summary()
print(regressor.input_shape)
print(regressor.output_shape)
print(X.shape)
print(t.shape)


# In[55]:


y = regressor(X)
y = np.expand_dims(y, axis = 2);
e = y - t


error_df = pd.DataFrame.from_dict(
    {
        'loss A': e[:,0,0],
        'loss C': e[:,1,0],
        'loss G': e[:,2,0],
        'loss T': e[:,3,0],
        'Quality (single aspect)': e[:,4,0],
    }
)
fig = px.scatter(error_df, y='loss A'); fig.show()


# In[45]:


fig = px.scatter(error_df, y='loss C'); fig.show()


# In[46]:


fig = px.scatter(error_df, y='loss G'); fig.show()


# In[47]:


fig = px.scatter(error_df, y='loss T'); fig.show()


# In[56]:


fig = px.scatter(error_df, y='Quality (single aspect)'); fig.show()


# In[49]:


# Weights
w = regressor.get_weights()
print(w[0])
print(w[1])
print(w[2])


# In[54]:


# Error terms
error_df.round().max()


# ## [Preliminary] Outcomes:
# ***
# #### [1] The Long-Short Term Memory Artificial Neural Network (LSTM NN)  is able to approximate perfectly the Sequence- and at least one aspect/flag of quality.
# ***
# #### [2] The following limitation is not valide from the higher oreder considerations:
# - We use only the  𝑁=500  past observations of the data. No {Mutations, Insertions, inversions, etc.} flags are used in the current implementation.
# 
# #### The use of the{Mutations, Insertions, inversions, etc.} flags to be specified.

# In[ ]:




