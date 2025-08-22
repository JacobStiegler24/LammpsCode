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

# Constants
timestep = 0.005 # lj
nevery = 100 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5E-9 # m
epsilon = 2.25E-20 # J
mass = 2.12E-22 # kg
F_LJ = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

k_theta = 7.785
k_r = 43.87
k_r_prime = 100
r_0 = 1
r_0_prime = 4

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F_LJ} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')

def Zr_bondF(r, k, F=0):
    alpha = (r*F)/(T_LJ)
    if F != 0:
        exponent = (-k_r/(2*T_LJ))*(r-r_0)**2
        integrand = r**2*np.exp(exponent)*(2*np.pi*np.sinh(alpha)/alpha)
    else:
        exponent = (-k_r/(2*T_LJ))*(r-r_0)**2
        integrand = r**2*np.exp(exponent)
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def Zr_bondU(r, k, F):
    alpha = (r*F)/(T_LJ)
    if F != 0:
        integrand = r**2*(((1-(r/r_0_prime)**2)**(2*k/T_LJ)))*(2*np.pi*np.sinh(alpha)/alpha)
    else:
        integrand = r**2*(((1-(r/r_0_prime)**2)**(2*k/T_LJ)))
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def main():
    counter = 0
    for F in range(0,60,20):
        q = np.linspace(0.01,r_0_prime,1000)
        # R_var = np.array(VarR_U(k_r_prime, F))
        # R_mean = np.array(ExpectedR_U(k_r_prime, F))
        norm_U, err = integrate.quad(Zr_bondU, 0, r_0_prime, args=(k_r_prime,F))
        norm_F, err = integrate.quad(Zr_bondF, 0, r_0 * 5, args=(k_r,F))
        print('####################')
        print(norm_F)
        print('####################')
        Z_U = np.array(Zr_bondU(q,k_r_prime,F)/norm_U)
        Z_F = np.array(Zr_bondF(q,k_r_prime,F)/norm_F)
        plt.figure(counter)
        plt.title(f'force {F}')
        plt.plot(q,Z_U, label='Unfolded')
        plt.plot(q,Z_F, label='folded')
        plt.xlabel('bond length (lj)')
        plt.ylabel('frequency')
        plt.legend()
        counter+=1
    plt.show()

main()