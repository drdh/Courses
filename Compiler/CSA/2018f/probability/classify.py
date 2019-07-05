#!/usr/bin/python3
import pymc3 as pm
import math
from data import data, X, Y
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

model = pm.Model()

with model:
    # xs = pm.Normal('xs', 0, 1, total_size=4, observed=X)
    xs = pm.Normal('xs', mu=0, sd=1, shape=4, observed=X)

    ws = pm.Normal('ws', mu=0, sd=1, shape=4)
    b = pm.Normal('b',mu=0, sd=1)

    p = pm.math.sigmoid(pm.math.dot(xs,ws)+b)

    y = pm.Bernoulli('y', p, observed=Y)
    # step = pm.Slice()
    # mcmc = pm.
    mcmc = pm.find_MAP(model=model)
    # mcmc = pm.sample(50000, step=step)
    
    print(mcmc)
    # pm.traceplot(mcmc)