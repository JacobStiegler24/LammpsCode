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

def ZTheta(theta, k_t):
    exponent = (-k_t/(2*T_LJ))*(theta-np.pi)**2
    integrand =  np.sin(theta)*np.exp(exponent)
    return integrand

def Zr_bond(r, k_r):
    exponent = (-k_r/(2*T_LJ))*(r-1)**2
    integrand = r**2*np.exp(exponent)

    # Rmean = ( (3/(2*k_r))+1)*1 / ( (1/(2*k_r))+1 )
    # STD=np.sqrt(1/(2*k_r))
    # integrand = 1/np.sqrt(2*3.14*STD**2)*np.exp(-(r-Rmean)**2/(2*STD**2))

    return integrand

def ExpectedThetaIntegral(theta, k_t):
    return theta*ZTheta(theta,k_t)

def ExpectedTheta(k_t):
    Z, Zerr = integrate.quad(ZTheta, 0, np.pi, args=(k_t,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedThetaIntegral, 0, np.pi, args=(k_t,))
    expectedTheta = invZ*integral
    expectedTheta_err = expectedTheta*(invZerr/invZ + err/integral)
    return expectedTheta, expectedTheta_err

def ExpectedRIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r

def ExpectedRSqrIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r**2

def ExpectedR(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, np.inf, args=(k_r,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bond, 0, np.inf, args=(k_r,))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedRSqr(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, np.inf, args=(k_r,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, np.inf, args=(k_r,))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, np.inf, args=(k_r,))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k_r):
    
    expectedR = ExpectedR(k_r)
    expectedRSqr = ExpectedRSqr(k_r)

    var = expectedRSqr-expectedR**2
    return var

def main():
    x = np.linspace(1, 100, 10000)
    t = np.linspace(1, 100, 10000)
    R_var = np.array([VarR(i) for i in x])
    Theta_exp = np.array([ExpectedTheta(i) for i in t])[:,0]

    R_var_ratio = R_var/2
    Theta_exp_ratio = (np.pi-Theta_exp)/np.pi

    Delta_R = np.abs(R_var_ratio - 0.1)
    Delta_Theta = np.abs(Theta_exp_ratio - 0.1)

    print(f'Best k_r = {x[Delta_R.argmin()]}')
    print(f'Best k_theta = {t[Delta_Theta.argmin()]}')

    plt.figure(1)
    plt.title('Delta Sigma_R/2r_0 - 0.1')
    plt.plot(x, Delta_R)
    plt.figure(2)
    plt.title('Delta (pi - <Theta>)/pi - 0.1')
    plt.plot(t, Delta_Theta)

    plt.show()

main()