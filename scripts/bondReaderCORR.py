print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
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
    path = os.path.join(os.getcwd(), r'projects/Fibrin-Monomer/output/CORR')
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
            "K_r": k_r/2,
            "K_theta": k_t/2,
            "types": np.array(btype_lst, dtype=int),
            "lengths": mlen_arr,
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

def main():
    data_df = readfile()
    data_df.sort_values(by=['K_r', 'K_theta'], inplace=True)
    
    rt_pairs = data_df[['K_r', 'K_theta']].drop_duplicates().values
    
    sigma_r = []
    sigma_t = []
    sigma_r_sem = []
    sigma_t_sem = []

    for k_r, k_t in rt_pairs:
        try:
            sigma_r.append(np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0])/2)
            sigma_r_sem.append((np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0])/2)/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['lengths'].iloc[0])))
            sigma_t.append(np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])/180)
            sigma_t_sem.append((np.std(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])/180)/np.sqrt(len(data_df[(data_df['K_r'] == k_r) & (data_df['K_theta'] == k_t)]['angle'].iloc[0])))
        except ValueError as e:
            print(f"Error calculating std for k_r={k_r}, k_t={k_t}: {e}")
            sigma_r.append(np.nan)
            sigma_t.append(np.nan)
            
    sigma_r_arr = np.array(sigma_r, dtype=np.float64)
    sigma_t_arr = np.array(sigma_t, dtype=np.float64)
    delta_r = np.abs(sigma_r_arr-0.1)
    delta_t = np.abs(sigma_t_arr-0.1)
    delta = delta_r**2 + delta_t**2

    print(f'index closest to 10%: {delta.argmin()}')
    print(f'sigma_r: {sigma_r_arr[delta.argmin()]}, sigma_t: {sigma_t_arr[delta.argmin()]}')
    print(f'k_r: {rt_pairs[delta.argmin()][0]}, k_t: {rt_pairs[delta.argmin()][1]}')

    k_r = data_df['K_r'].iloc[np.where(delta < 0.001)]
    k_t = data_df['K_theta'].iloc[np.where(delta < 0.001)]

    corr = np.corrcoef((sigma_r), np.log(sigma_t))
    print("Correlation coefficient:", corr)

    os.chdir('..')
    os.chdir('..')
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    monomer_df = pd.DataFrame({
        "K_r": data_df['K_r'],
        "K_theta": data_df['K_theta'],
        "sigma_r": sigma_r,
        "sigma_r_sem": sigma_r_sem,
        "sigma_t": sigma_t,
        "sigma_t_sem": sigma_t_sem,
        "lengths": data_df['lengths'].values,
        "angles": data_df['angle'].values,
    })

    # plt.figure(0)
    # r_vals = data_df[['K_r']].drop_duplicates().values.flatten()    
    # norm = plt.Normalize(vmin=min(r_vals), vmax=max(r_vals))
    # cmap = plt.get_cmap('plasma')
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # for k_r in r_vals:
    #     plt.plot(
    #         monomer_df[monomer_df['K_r'] == k_r]['K_theta'],
    #         monomer_df[monomer_df['K_r'] == k_r]['sigma_r'],
    #         c=cmap(norm(k_r))
    #     )
    # plt.colorbar(sm, label='K_r')
    # plt.xlabel('K_theta')
    # plt.ylabel('Sigma_r / 2 lj (molecule)')
    # plt.legend()
    # plt.savefig('sigmar_kt_line.png', dpi=1000)

    # plt.figure(1)
    # t_vals = data_df[['K_theta']].drop_duplicates().values.flatten()
    # norm = plt.Normalize(vmin=min(t_vals), vmax=max(t_vals))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # for k_t in t_vals:
    #     plt.plot(
    #         monomer_df[monomer_df['K_theta'] == k_t]['K_r'],
    #         monomer_df[monomer_df['K_theta'] == k_t]['sigma_t'],
    #         c=cmap(norm(k_t))
    #     )
    # plt.colorbar(sm, label='K_theta')
    # plt.xlabel('K_r')
    # plt.ylabel('Sigma_theta / 180')
    # plt.legend()
    # plt.savefig('sigmat_kr_line.png', dpi=1000)

    
    plt.figure(2)
    plt.yscale('log')
    plt.xscale('log')
    x = np.linspace(1, 60, 200)
    y = np.sqrt(T_LJ/(8*(x)))
    plt.xlim(2,60)
    plt.ylim(0.04,0.2)
    plt.plot(x, y, color='red')

    plt.errorbar(
        monomer_df['K_r'],
        monomer_df['sigma_r'],
        yerr=monomer_df['sigma_r_sem'],
        color='black',
        fmt='.',   
        markersize=1,
        capsize=2,
        capthick=0.5,
        elinewidth=0.3
    )
    plt.scatter(
        monomer_df['K_r'],
        monomer_df['sigma_r'],
        c=monomer_df['K_theta'],
        s=10
    )
    plt.colorbar(label='K_theta')
    plt.xlabel('K_r')
    plt.ylabel('Sigma_r / 2 lj (molecule)')
    plt.title('Sigma_r / monomer R_0 coloured by K_theta (logxy)')
    plt.legend()
    plt.savefig('sigma_k_r.png', dpi=1000)

    plt.figure(3)
    plt.yscale('log')
    plt.xscale('log')
    x = np.linspace(0.5, 12, 100)
    y = np.sqrt(T_LJ/(x))/(np.sqrt(6)*np.pi)
    plt.plot(x, y, color='red')

    plt.errorbar(
        monomer_df['K_theta'],
        monomer_df['sigma_t'],
        yerr=monomer_df['sigma_t_sem'],
        color='black',
        fmt='.',      # adds point markers
        markersize=1,
        capsize=2,
        capthick=0.5,
        elinewidth=0.3
    )
    plt.scatter(
        monomer_df['K_theta'],
        monomer_df['sigma_t'],
        c=monomer_df['K_r'],
        s=10
    )
    plt.colorbar(label='K_r')
    plt.xlabel('K_theta')
    plt.ylabel('Sigma_theta / 180')
    plt.title('Sigma_theta / theta_0 coloured by K_r (logxy)')
    plt.legend()
    plt.savefig('sigma_k_t.png', dpi=1000)

    plt.show()

    monomer_df.to_csv('corr_data.csv')
main()