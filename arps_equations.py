# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:55:59 2022

@author: gamaliel.adun
"""
import numpy as np
import streamlit as st

@st.cache_data
def ArpsRateExp(t,Qi,Di):
    """Exponential Case: b=0
    
    t: Production time
    Qi: Initial production rate
    Di: Inital decline rate
    b: Arps empirical exponent
    """
    return Qi * np.exp(-1*Di*t)

@st.cache_data
def ArpsRateHar(t,Qi,Di):
    """Harmonic Case: b=1
    
    t: Production time
    Qi: Initial production rate
    Di: Inital decline rate
    b: Arps empirical exponent
    """
    return Qi/(1+(Di*t))

@st.cache_data
def ArpsRateHyp(t,Qi,Di,b):
    """Hyperbolic Case: 0 <= b <= 1
    
    t: Production time
    Qi: Initial production rate
    Di: Inital decline rate
    b: Arps empirical exponent
    """
    return Qi / ((1+(b*Di*t))**(1/b))        

@st.cache_data
def NominalDecline(Di,t,b):
    """Calculates the Arps Nominal Decline 
    
    t: Production time
    Di: Inital decline rate
    b: Arps empirical exponent
    """
    
    if b < 0 or b > 1:
        raise ValueError('Error: b must be between 0 and 1')
    else:
        return Di / (1+(b*Di*t))

@st.cache_data
def ArpsCumProd(Qc,Qel,D,b):    
    """Calculates the Arps Cummulative Production
   
    Qc: Current production rate at beginning of forecast
    Qel: Production rate at economic limit
    D: Inital decline rate at beginning of forecast
    b: Arps empirical exponent
    """
    
    if b < 0 or b > 1:
        raise ValueError('Error: b must be between 0 and 1')
    elif b == 1:
        return (Qc/D) * np.log10(Qc/Qel)
    else:
        return (Qc**b)*((Qc**(1-b))-(Qel**(1-b)))/(D*(1-b))