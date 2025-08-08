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
timestep = 0.0001 # lj
nevery = 5 # how often data is dumped
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
def readfile():
    # Change directory to the project folder
    print(os.getcwd())
    os.chdir('..')
    print(os.getcwd())
    path = os.path.join(os.getcwd(), r'Fibrin-Monomer\output\CORR')
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
        r_match = re.search(r'k_r(\d+(?:\.\d+)?)', folder)
        t_match = re.search(r'k_theta(\d+(?:\.\d+)?)', folder)

        if not (r_match and t_match and seed_match):
            continue
        
        seed = int(seed_match.group(1))
        k_r = float(r_match.group(1))
        k_t = float(t_match.group(1))

        folder_path = os.path.join(os.getcwd(), folder)
        os.chdir(folder_path)

        btype_lst = [[],[]]
        blen_lst = [[],[]]
        
        with open('bondlengths.dump') as f:
            isItem = False
            counter2 = 0
            for line in f:
                if isItem:
                    btype_lst[counter2].append(line[0])
                    blen_lst[counter2].append(line[2::].strip(' \n'))
                    counter2 += 1
                if counter2 == 2:
                    isItem = False
                    counter2 = 0
                if 'ENTRIES c_btype c_bondlen' in line:
                    isItem = True
        
        bangle_lst = []
        with open('angles.dump') as f:
            isItem = False
            for line in f:
                if isItem:
                    isItem = False
                    bangle_lst.append(line)
                if 'ITEM: ENTRIES c_myAngle' in line:
                    isItem = True

        mlen_arr = np.array(blen_lst, dtype=np.float64)[0, :] + np.array(blen_lst, dtype=np.float64)[1, :]
        
        row = {
            "seed": seed,
            "K_r": k_r*2,
            "K_theta": k_t*2,
            "types": np.array(btype_lst, dtype=int),
            "lengths": mlen_arr,
            "bond lengths": np.array(blen_lst),
            "angle": np.array(bangle_lst, dtype=np.float64)
        }

        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)
    df1 = df.groupby(['K_r', 'K_theta']).agg({
        'lengths': lambda x: np.concatenate(x.to_numpy()),
        'angle': lambda x: np.concatenate(x.to_numpy()),
        'types': lambda x: np.concatenate(x.to_numpy())
    }).reset_index()

    # df2 = df.groupby(['K_r', 'K_theta']).agg({
    #     'K_r': 'first',
    #     'K_theta': 'first',
    #     'lengths': lambda x: np.mean(x),
    #     'angle': lambda x: np.mean(x),
    #     'types': lambda x: np.mean(x)
    # }).reset_index()
    return df1

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
    return integrand

def ExpectedThetaIntegral(theta, k_t):
    exponent = (-k_t/(2*T_LJ))*(theta-np.pi)**2
    integrand =  theta*np.sin(theta)*np.exp(exponent)
    return integrand

def ExpectedTheta(k_t):
    Z, Zerr = integrate.quad(ZTheta, 0, np.pi, args=(k_t,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedThetaIntegral, 0, np.pi, args=(k_t,))
    expectedTheta = invZ*integral
    expectedTheta_err = expectedTheta*(invZerr/invZ + err/integral)
    return expectedTheta, expectedTheta_err

def ExpectedRIntegral_bond(r, k_r):
    exponent = (-k_r/(2*T_LJ))*(r-1)**2
    integrand = r**3*np.exp(exponent)
    return integrand

def ExpectedRSqrIntegral_bond(r, k_r):
    exponent = (-k_r/(2*T_LJ))*(r-1)**2
    integrand = r**4*np.exp(exponent)
    return integrand

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

    integral, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, np.inf, args=(k_r,))
    expectedR_sqr_new = invZ*integral*2 + 2*(invZ*integral)**2

    return expectedR_sqr_new

def VarR(k_r):
    
    expectedR = ExpectedR(k_r)
    expectedRSqr = ExpectedRSqr(k_r)

    var = expectedRSqr-expectedR**2
    return var

def stats(data_df):
    rt_pairs = data_df[['K_r', 'K_theta']].drop_duplicates().values
    

    std_r = []
    sem_std_r = []
    mean_r = []
    sem_r = []
    mean_t = []
    sem_t = []
    std_t = []
    sem_std_t = []

    for k_r, k_t in rt_pairs:
        try:
            std_r.append(np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0]))
            sem_std_r.append((np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0]))/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0])))
            
            mean_r.append(np.mean(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0]))
            sem_r.append((np.mean(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0]))/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0])))

            std_t.append(np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])*np.pi/180)
            sem_std_t.append((np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])*np.pi/180)/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])))

            mean_t.append(np.mean(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])*np.pi/180)
            sem_t.append((np.mean(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])*np.pi/180)/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])))

        except ValueError as e:
            print(f"Error calculating std for k_r={k_r}, k_t={k_t}: {e}")
            std_r.append(np.nan)
            sem_std_r.append(np.nan)

            mean_r.append(np.nan)
            mean_r.append(np.nan)

            mean_t.append(np.nan)
            sem_t.append(np.nan)

    #mean_r_arr = np.array(mean_r, dtype=np.float64)
    mean_t_arr = np.array(mean_t, dtype=np.float64)

    #delta_r = np.abs(mean_r_arr-0.1)
    delta_t = np.abs(np.pi-mean_t_arr-0.1)
    #delta = delta_r**2 + delta_t**2

    print(f'index closest to 10%: {delta_t.argmin()}')
    #print(f'sigma_r: {sigma_r_arr[delta.argmin()]}, sigma_t: {sigma_t_arr[delta.argmin()]}')
    print(f'mean_t: {mean_t_arr[delta_t.argmin()]}')
    print(f'k_r: {rt_pairs[delta_t.argmin()][0]}, k_t: {rt_pairs[delta_t.argmin()][1]}')

    #k_r = data_df['K_r'].iloc[np.where(delta_t < 0.001)]
    #k_t = data_df['K_theta'].iloc[np.where(delta_t < 0.001)]

    #corr = np.corrcoef((sigma_r), np.log(sigma_t))
    #print("Correlation coefficient:", corr)

    monomer_df = pd.DataFrame({
        "K_r": data_df['K_r'],
        "K_theta": data_df['K_theta'],

        "r mean": mean_r,
        "r sem": sem_r,
        "r std": std_r,
        "r std sem": sem_std_r,

        "theta mean": mean_t_arr,
        "theta sem": sem_t,
        "theta std": std_t,
        "theta std sem": sem_std_t,

        "lengths": data_df['lengths'].values,
        "angles": data_df['angle'].values,
    })
    return monomer_df 

def main():
    data_df = readfile()
    data_df.sort_values(by=['K_r', 'K_theta'], inplace=True)
    
    monomer_df = stats(data_df)

    os.chdir('..')
    os.chdir('..')
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    plt.figure(1)
    plt.ylim(0,5)
    x = np.linspace(0.01, 20, 200)
    p = np.sqrt(np.array([VarR(i) for i in x]))
    print('sigma---------------------')
    print(p)
    l = (np.array([ExpectedR(i) for i in x]))
    print('<R>-----------------------')
    print(l)
    y = p/l
    plt.plot(x, l, label='theory <R>')
    plt.errorbar(
        monomer_df['K_r'],
        ((monomer_df['r mean'])),
        yerr=(monomer_df['r sem']),
        color='black',
        fmt='.',   
        markersize=1,
        capsize=2,
        capthick=0.5,
        elinewidth=0.3
    )
    plt.scatter(
        monomer_df['K_r'],
        (monomer_df['r mean']),
        c=monomer_df['K_theta'],
        s=10
    )
    plt.colorbar(label='K_theta')
    plt.xlabel('K_r')
    plt.ylabel('<R> (lj)')
    plt.title('Expected monomer length coloured by K_theta (logxy)')
    plt.legend()


    plt.figure(2)
    plt.ylim(0,5)
    plt.plot(x, l, label='theory std')

    plt.errorbar(
        monomer_df['K_r'],
        ((monomer_df['r std'])),
        yerr=(monomer_df['r std sem']),
        color='black',
        fmt='.',   
        markersize=1,
        capsize=2,
        capthick=0.5,
        elinewidth=0.3
    )
    plt.scatter(
        monomer_df['K_r'],
        (monomer_df['r std']),
        c=monomer_df['K_theta'],
        s=10
    )
    plt.colorbar(label='K_theta')
    plt.xlabel('K_r')
    plt.ylabel('r std (lj)')
    plt.title('std of monomer length coloured by K_theta (logxy)')
    plt.legend()

    plt.legend()
   
    plt.figure(3)
    x = np.linspace(0.1, 40, 200)
    y = np.array([ExpectedTheta(i) for i in x])
    plt.plot(x, (y[:,0]), color='red')

    plt.errorbar(
        monomer_df['K_theta'],
        monomer_df['theta mean'],
        yerr=monomer_df['theta sem'],
        color='black',
        fmt='.',      # adds point markers
        markersize=1,
        capsize=2,
        capthick=0.5,
        elinewidth=0.3
    )
    plt.scatter(
        monomer_df['K_theta'],
        (monomer_df['theta mean']),
        c=monomer_df['K_r'],
        s=10
    )
    plt.colorbar(label='K_r')
    plt.xlabel('K_theta')
    plt.ylabel('Mean theta (rad)')
    plt.title('Mean theta coloured by K_r (logxy)')
    plt.legend()
    plt.show()
    # monomer_df.to_csv('corr_data.csv')

main()