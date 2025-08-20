print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
from scipy import integrate
from math import floor, log10
print('imported')

# Point of script:
# One/two bond unfolded fractions
# Mean time one/two unfolded + std/se
# Free energy + standard error

# Constants
timestep = 0.005 # lj
nevery = 100 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5E-9 # m
epsilon = 2.25E-20 # J
mass = 2.12E-22 # kg
F = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')

def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def Zr_bond(r, k_r):
    integrand = r**2*((1-(r**2)/4)**(k_r*2))
    return integrand

def ExpectedRIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r

def ExpectedRSqrIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r**2

def ExpectedR(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k_r,))
    try:
        invZ = 1/Z
        invZerr = Zerr/(Z**2)
    except ZeroDivisionError:
        invZ = 10E4
        invZerr = 10E4
    
    integral , err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k_r,))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedRSqr(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k_r,))
    try:
        invZ = 1/Z
        invZerr = Zerr/(Z**2)
    except ZeroDivisionError:
        invZ = 10E4
        invZerr = 10E4

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, 2, args=(k_r,))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k_r,))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k_r):
    
    expectedR = ExpectedR(k_r)
    expectedRSqr = ExpectedRSqr(k_r)

    var = expectedRSqr-expectedR**2
    return var

def main():
    x = np.linspace(1, 100, 1000)
    t = np.linspace(0.01,10,10)
    q = np.linspace(0.01,2,1000)
    R_var = np.array([VarR(i) for i in x])
    R_mean = np.array([ExpectedR(i) for i in x])
    Z = np.array([[Zr_bond(i,j)/integrate.quad(Zr_bond, 0, 2, args=(j)) for i in q] for j in t])

    plt.figure(1)
    plt.title('Sigma_r')
    plt.plot(x, R_var)
    plt.figure(2)
    plt.title('<r>')
    plt.plot(x, R_mean)
    plt.figure(3)
    plt.ylim(0,5)
    plt.xlim(0,2)
    for row_idx, k_r in enumerate(t):
        plt.plot(q, Z[row_idx, :], label=fr"$k_r = {k_r}$")
    plt.legend()
    plt.figure(4)
    y = np.array([[-0.5*i*4*np.log(1-(j/2)**2) for j in q] for i in t])
    for row_idx, k_r in enumerate(t):
        plt.plot(q, y[row_idx, :], label=fr"$k_r = {k_r}$")
    plt.show()

main()