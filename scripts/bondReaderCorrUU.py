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

def Zr_bond(r, k, F):
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

def ExpectedRIntegral_bond(r, k, F):
    return Zr_bond(r,k,F)*r

def ExpectedRSqrIntegral_bond(r, k, F):
    return Zr_bond(r,k,F)*r**2

def ExpectedR(k,F):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedRSqr(k,F):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, 2, args=(k,F))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k,F):
    
    expectedR = ExpectedR(k,F)
    expectedRSqr = ExpectedRSqr(k,F)

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
    data_df = readfile()
    data_df.sort_values(by=['force'], inplace=True)
    
    # monomer_df = stats(data_df)

    os.chdir('..')
    os.chdir('..')
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    print(data_df)
    plt.figure(1)

    intlim = [0,2]

    fig, ax = plt.subplots()
    ax.set_box_aspect(2/3)
    x = np.linspace(0,2,200)


    for i in range(0,5):
        F = data_df['force'].iloc[i]
        r = np.array([Zr_bond(i, k_r_prime, F) for i in x])
        Z, err = integrate.quad(Zr_bond, intlim[0], intlim[1], args=(k_r_prime, F))
        plt.plot(x, r/Z, label=f"force = {F} lj theory")  
        plt.hist(data_df['bond lengths'].iloc[i][0,:], bins=200, density=True, alpha=0.5, label=f"force = {F} lj")

    plt.xlabel('Bond 1 length (lj)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('UU_bond1_length_histogram_force.eps', bbox_inches='tight')


    plt.figure(2)

    intlim = [0,2]

    fig, ax = plt.subplots()
    ax.set_box_aspect(2/3)
    x = np.linspace(0,2,200)

    for i in range(0,5):
        F = data_df['force'].iloc[i]
        print(f'force {F}')
        r = np.array([Zr_bond(i, k_r_prime, F) for i in x])
        Z, err = integrate.quad(Zr_bond, intlim[0], intlim[1], args=(k_r_prime, F))
        plt.plot(x, r/Z, label=f"force = {F} lj theory")  
        plt.hist(data_df['bond lengths'].iloc[i][1,:], bins=200, density=True, alpha=0.5, label=f"force = {F} lj")
        
    plt.xlabel('Bond 2 length (lj)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('UU_bond2_length_histogram_force.eps', bbox_inches='tight')


    plt.show()

main()