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
F_LJ = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

k_theta = 7.785
k_r = 43.87
k_r_prime = 3

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F_LJ} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')
def readfile():
    # Change directory to the project folder
    print(f'current dir: {os.getcwd()}')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'output/ForceSeedUU')
    # path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\CORRuu'
    os.chdir(path)
    print(os.getcwd())

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        # output/k_r${k_r}_k_theta${k_theta}
        seed_match = re.search(r'Seed(\d+)?', folder)
        r_match = re.search(r'Force(\d+(?:\.\d+)?)', folder)

        if not (r_match and seed_match):
            continue
        
        seed = int(seed_match.group(1))
        force = float(r_match.group(1))

        folder_path = os.path.join(os.getcwd(), folder)
        os.chdir(folder_path)

        btype_lst = [[],[]]
        blen_lst = [[],[]]
        with open('bondlengths.dump') as f:
            isItem = False
            counter2 = 0
            for line in f:
                if 'ENTRIES c_btype c_bondlen' in line:
                    isItem = True
                    counter2 = 0
                    continue
                if not isItem:
                    continue
                if counter2 == 2:
                    isItem = False
                    counter2 = 0
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    btype_lst[counter2].append(int(parts[0]))
                    blen_lst[counter2].append(float(parts[1]))
                counter2 += 1
                
        bangle_lst = []
        with open('angles.dump') as f:
            isItem = False
            for line in f:
                if 'ITEM: ENTRIES c_myAngle' in line:
                    isItem = True
                    continue
                if isItem:
                    try:
                        bangle_lst.append(float(line.strip()))
                    except ValueError:
                        pass
                    isItem = False
                

        mlen_arr = np.array(blen_lst, dtype=np.float64)[0, :] + np.array(blen_lst, dtype=np.float64)[1, :]
        blen_arr = np.array(blen_lst, dtype=float) 
        row = {
            "seed": seed,

            "force": force,

            "types": np.array(btype_lst, dtype=int),

            "bond lengths": blen_arr,       
            "monomer lengths": mlen_arr,       

            "angle_deg": np.array(bangle_lst, dtype=float),
        }
        
        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)

    print(df)

    df1 = df.groupby(['force']).agg({
        'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
        'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
        'types':         lambda s: np.concatenate(s.to_numpy()),
        'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
    }).reset_index()
    return df1

def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def Zr_bondF(r, k, F=0):
    alpha = (r*F)/(T_LJ)
    if F != 0:
        exponent = (-k_r/(2*T_LJ))*(r-1)**2
        integrand = r**2*np.exp(exponent)*(2*np.pi*np.sinh(alpha)/alpha)
    else:
        exponent = (-k_r/(2*T_LJ))*(r-1)**2
        integrand = r**2*np.exp(exponent)
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def Zr_bondU(r, k, F):
    alpha = (r*F)/(T_LJ)
    if F != 0:
        integrand = r**2*(((1-(r**2)/4)**(2*k/T_LJ)))*(2*np.pi*np.sinh(alpha)/alpha)
    else:
        integrand = r**2*(((1-(r**2)/4)**(2*k/T_LJ)))
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def U_eff(r,k,F):
    alpha = (r*F)/(2*T_LJ)
    return -(2*k/T_LJ)*np.log(1-(r**2)/4)-np.log((2*np.pi*np.sinh(alpha)/alpha))

def Zr_monomer(r, k):
    integrand = r**2*((1-(r**2)/16)**(k*8))
    return integrand

def ExpectedRIntegral_bondU(r, k, F):
    return Zr_bondU(r,k,F)*r

def ExpectedRIntegral_bondF(r, k, F):
    return Zr_bondF(r,k,F)*r

def ExpectedRSqrIntegral_bondU(r, k, F):
    return Zr_bondU(r,k,F)*r**2

def ExpectedRSqrIntegral_bondF(r, k, F):
    return Zr_bondF(r,k,F)*r**2

def ExpectedR_U(k,F):
    Z, Zerr = integrate.quad(Zr_bondU, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bondU, 0, 2, args=(k,F))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedR_F(k,F):
    Z, Zerr = integrate.quad(Zr_bondF, 0, np.inf, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bondF, 0, np.inf, args=(k,F))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedRSqr_U(k,F):
    Z, Zerr = integrate.quad(Zr_bondU, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bondU, 0, 2, args=(k,F))
    integral2, err = integrate.quad(ExpectedRIntegral_bondU, 0, 2, args=(k,F))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2
    return expectedR_sqr_new

def ExpectedRSqr_F(k,F):
    Z, Zerr = integrate.quad(Zr_bondF, 0, np.inf, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bondF, 0, 2, args=(k,F))
    integral2, err = integrate.quad(ExpectedRIntegral_bondF, 0, 2, args=(k,F))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2
    return expectedR_sqr_new

def VarR_U(k,F):
    
    expectedR = ExpectedR_U(k,F)
    expectedRSqr = ExpectedRSqr_U(k,F)

    var = expectedRSqr-expectedR**2
    return var

def VarR_F(k,F):
    
    expectedR = ExpectedR_F(k,F)
    expectedRSqr = ExpectedRSqr_F(k,F)

    var = expectedRSqr-expectedR**2
    return var

def stats(data_df):
    rt_pairs = data_df['force'].drop_duplicates().values
    

    std_mlen = []
    sem_std_mlen = []
    mean_mlen = []
    sem_mlen = []

    mean_blen = []

    for force in rt_pairs:
        try:
            std_mlen.append(np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
            sem_std_mlen.append((np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
            
            mean_mlen.append(np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
            sem_mlen.append((np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
            mean_blen.append(np.max(np.array(data_df[data_df['force'] == force]['bond lengths'].iloc[0]), axis=1))

        except ValueError as e:
            print(f"Error calculating std for f={force}: {e}")
            std_mlen.append(np.nan)
            sem_std_mlen.append(np.nan)

            mean_mlen.append(np.nan)
            mean_mlen.append(np.nan)


    monomer_df = pd.DataFrame({
        "force": data_df['force'],

        "monomer length mean": mean_mlen,
        "monomer length sem": sem_mlen,
        "monomer length std": std_mlen,
        "monomer length std sem": sem_std_mlen,

        "blen mean": mean_blen,

        "lengths": data_df['monomer lengths'].values,
        "angles": data_df['angle_deg'].values,
    })
    
    return monomer_df 

def main():
    counter = 0
    for F in range(0,60,10):
        q = np.linspace(0.01,2,1000)
        # R_var = np.array(VarR_U(k_r_prime, F))
        # R_mean = np.array(ExpectedR_U(k_r_prime, F))
        norm_U, err = integrate.quad(Zr_bondU, 0, 2, args=(k_r_prime,F))
        norm_F, err = integrate.quad(Zr_bondF, 0, 2, args=(k_r_prime,F))
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