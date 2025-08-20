print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
from scipy.stats import sem
from math import floor, log10
print('imported')

# Point of script:
# One/two bond unfolded fractions
# Mean time one/two unfolded + std/se
# Free energy + standard error

# Constants
timestep = 0.000001 # lj
nevery = 100 # how often data is dumped
run_no = 50000000
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5E-9 # m
epsilon = 2.25E-20 # J
mass = 2.12E-22 # kg
F = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s



print(f'Simulation time: {timestep*run_no*tau}')
print(f'timestep: {timestep*tau}')
print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')
def readfile():
    # Change directory to the project folder
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'output/ForceSeed')
    #path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\ForceSeed'
    os.chdir(path)
    print(os.getcwd())

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        force_match = re.search(r'Force(\d+(?:\.\d+)?)', folder)
        seed_match = re.search(r'Seed(\d+)', folder)

        if not (force_match and seed_match):
            continue

        force = float(force_match.group(1))
        seed = int(seed_match.group(1))

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

        row = {
            "force": force,
            "seed": seed,
            "types": np.array(btype_lst, dtype=int),
            "lengths": np.array(blen_lst, dtype=np.float64),
            "angle": np.array(bangle_lst, dtype=np.float64)
        }

        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)

    return df

def compute(btype_arr, blen_arr, bangle_arr):
    # Calculate:
    # Mean extension
    # One/two bond unfolded fractions
    # Mean time one/two unfolded + std/se
    # Free energy + standard error

    # mean_extension = np.mean(blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0])
    # extension_std = np.std(blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0])
    # extension_sem = extension_std / np.sqrt(len(blen_arr[0]))

    length_mean = np.mean(blen_arr[0]+blen_arr[1])
    length_std = np.std(blen_arr[0]+blen_arr[1])
    length_sem = length_std / np.sqrt(len(blen_arr[0]))

    mean_angle = np.mean(bangle_arr)* np.pi / 180
    angle_std = np.std(bangle_arr) * np.pi / 180
    angle_sem = angle_std / np.sqrt(len(bangle_arr))
    
    #######################################################################################
    # unfolded fractions
    timestep_no = len(btype_arr[1])

    UU_count = 0
    FU_count = 0
    for i in range(timestep_no):
        if btype_arr[0][i] == 2 and btype_arr[1][i] == 2:
            UU_count += 1
        elif btype_arr[0][i] == 2 or btype_arr[1][i] == 2:
            FU_count += 1

    UU_fraction = UU_count / btype_arr.shape[1]
    FU_fraction = FU_count / btype_arr.shape[1]
    FF_fraction = 1 - (UU_fraction + FU_fraction)

    #######################################################################################
    # unfold times
    FU_time_lst = []
    UU_time_lst = []
    counter = 0
    FU = False
    UU = False
    for i in range(timestep_no):
        if btype_arr[0][i] != btype_arr[1][i]:
            if not FU and not UU:
                FU = True
                counter = 0
            elif UU:
                FU = True
                UU = False
                UU_time_lst.append(counter)
                counter = 0

        elif btype_arr[0][i] == 2 and btype_arr[1][i] == 2:
            if not FU and not UU:
                UU = True
                counter = 0
            elif FU:
                UU = True
                FU = False
                FU_time_lst.append(counter)
                counter = 0

        elif btype_arr[0][i] == 1 and btype_arr[1][i] == 1:
            if FU:
                FU = False
                FU_time_lst.append(counter)
                counter = 0
            elif UU:
                UU = False
                UU_time_lst.append(counter)
                counter = 0
        
        if FU or UU:
            counter += 1

    if FU:
        FU_time_lst.append(counter)
    elif UU:
        UU_time_lst.append(counter)

    FU_time_arr = np.array(FU_time_lst, dtype= float)
    UU_time_arr = np.array(UU_time_lst, dtype= float)

    # if no values in the list, sem will be zero. Set sem to timestep to avoid division by 
    # zero, this should be physically meaningful.
    if len(FU_time_arr) > 0:
        FU_mean_duration = np.mean(FU_time_arr * indextime)
        FU_mean_duration_std = np.std(FU_time_arr * indextime)
        FU_mean_duration_se = FU_mean_duration_std / np.sqrt(len(FU_time_lst))
    else:
        FU_mean_duration = 0
        FU_mean_duration_std = 0
        FU_mean_duration_se = timestep

    if len(UU_time_arr) > 0:
        UU_mean_duration = np.mean(UU_time_arr * indextime)
        UU_mean_duration_std = np.std(UU_time_arr * indextime)
        UU_mean_duration_se = UU_mean_duration_std / np.sqrt(len(UU_time_lst))
    else:
        UU_mean_duration = 0
        UU_mean_duration_std = 0
        UU_mean_duration_se = timestep

    ############################################################################################
    # Equilibrium constant and free energy calculations
    # K_eq = (fraction of folded bs) / (fraction of unfolded bs)
    # Free energy G/kT = ln(K_eq). 
    # This is the difference in free energy between folded and unfolded states.

    if FF_fraction > 0:
        K_eq_UU = UU_fraction / FF_fraction
        K_eq_FU = FU_fraction / FF_fraction
    else:
        K_eq_FU = 10e40
        K_eq_UU = 10e40
    return length_mean, length_std, length_sem, mean_angle, angle_std, angle_sem, FU_fraction, UU_fraction, FU_mean_duration, FU_mean_duration_std, FU_mean_duration_se,\
        UU_mean_duration, UU_mean_duration_std, UU_mean_duration_se, K_eq_FU, K_eq_UU

def weighted_avg_std_se(group, mean_col, sem_col, std_col):
    # weighted average and standard error of the mean
    # pooled standard deviation. https://www.statisticshowto.com/pooled-standard-deviation

    mean = group[mean_col].values
    std = group[std_col].values
    sem = group[sem_col].values

    valid = (~np.isnan(mean)) & (~np.isnan(std)) & (sem > 0) & (~np.isnan(sem))
    if np.any(~valid):
        print(f"Valid mean/std/sem NaN or sem > 0 entries: {np.sum(valid)} out of {len(valid)}")

    if np.sum(valid) == 0:
        return pd.Series({'mean': np.nan, 'std': np.nan, 'sem': np.nan})
    
    mean = mean[valid]
    std = std[valid]
    sem = sem[valid]

    weights = 1 / sem**2
    weighted_mean = np.sum(weights * mean) / np.sum(weights)
    weighted_sem = np.sqrt(1 / np.sum(weights))
    mean_std = np.mean(std)  # pooled standard deviation
    mean_std_sem = np.std(std) / np.sqrt(2*len(std)-2)

    return pd.Series({'mean': weighted_mean, 'std': mean_std, 'std_sem': mean_std_sem, 'sem': weighted_sem})

def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def computeSeedAveraged(data_df):
    # Equilibrium constant and free energy calculations
    # K_eq = (fraction of folded bs) / (fraction of unfolded bs)
    # Free energy G/kT = ln(K_eq). 
    # This is the difference in free energy between folded and unfolded states.

    
    data_force_groups = data_df.groupby(['force'], group_keys=True)
    
    analysis_df = data_force_groups.agg({
        'FU fraction': ['mean', 'std', 'sem'],
        'UU fraction': ['mean', 'std', 'sem'],
        'FU equilibrium constant': ['mean', 'std', 'sem'],
        'UU equilibrium constant': ['mean', 'std', 'sem'],
    })

    extension = data_df.groupby('force').apply(
        lambda group: weighted_avg_std_se(
            group, 'mean extension', 'extension sem', 'extension std'
        )
    )

    angle = data_df.groupby('force').apply(
        lambda group: weighted_avg_std_se(
            group, 'angle', 'angle sem', 'angle std'
        )
    )

    FU_duration = data_df.groupby('force').apply(
        lambda group: weighted_avg_std_se(
            group, 'FU mean duration', 'FU mean duration sem', 'FU mean duration std'
        )
    )
    UU_duration = data_df.groupby('force').apply(
            lambda group: weighted_avg_std_se(
                group, 'UU mean duration', 'UU mean duration sem', 'UU mean duration std'
        )
    )

    analysis_df[('extension', 'mean')] = extension['mean']
    analysis_df[('extension', 'std')] = extension['std']
    analysis_df[('extension', 'std_sem')] = extension['std_sem']  
    analysis_df[('extension', 'sem')] = extension['sem']
    analysis_df[('angle', 'mean')] = angle['mean']
    analysis_df[('angle', 'std')] = angle['std']
    analysis_df[('angle', 'std_sem')] = angle['std_sem']
    analysis_df[('angle', 'sem')] = angle['sem']
    analysis_df[('FU mean duration', 'mean')] = FU_duration['mean']
    analysis_df[('FU mean duration', 'std')]  = FU_duration['std']
    analysis_df[('FU mean duration', 'sem')]   = FU_duration['sem']
    analysis_df[('UU mean duration', 'mean')] = UU_duration['mean']
    analysis_df[('UU mean duration', 'std')]  = UU_duration['std']
    analysis_df[('UU mean duration', 'sem')]   = UU_duration['sem']

    eq_const = analysis_df[('FU equilibrium constant', 'mean')]
    sem = analysis_df[('FU equilibrium constant', 'sem')]

    fe_val = np.where(eq_const > 0, -np.log(eq_const), np.inf)
    fe_se = np.where(eq_const > 0, sem / eq_const, np.nan)

    analysis_df[('FU free energy', 'value')] = fe_val
    analysis_df[('FU free energy', 'sem')]    = fe_se

    eq_const = analysis_df[('UU equilibrium constant', 'mean')]
    sem = analysis_df[('UU equilibrium constant', 'sem')]

    fe_val = np.where(eq_const > 0, -np.log(eq_const), np.inf)
    fe_se = np.where(eq_const > 0, sem / eq_const, np.nan)

    analysis_df[('UU free energy', 'value')] = fe_val
    analysis_df[('UU free energy', 'sem')]    = fe_se

    return analysis_df

def main():
    data_df = readfile()
    rows = []
    data_df.sort_values(by=['force', 'seed'], inplace=True)
    force_seed_pairs = data_df[['force', 'seed']].drop_duplicates().values
    for force, seed in force_seed_pairs:
        
        mean_extension, extension_std, extension_sem, mean_angle, angle_std, angle_sem, FU_fraction, UU_fraction, FU_mean_duration, FU_mean_duration_std, \
        FU_mean_duration_se, UU_mean_duration, UU_mean_duration_std, UU_mean_duration_se, \
        K_eq_FU, K_eq_UU \
        = compute(data_df.loc[(data_df['force'] == force) & (data_df['seed'] == seed)]['types'].iloc[0], \
                data_df.loc[(data_df['force'] == force) & (data_df['seed'] == seed)]['lengths'].iloc[0], \
                    data_df.loc[(data_df['force'] == force) & (data_df['seed'] == seed)]['angle'].iloc[0]
                )

        row = {
        "force": force, # * F
        "seed": seed,
        "mean extension": mean_extension * sigma,
        "extension std": extension_std * sigma,
        "extension sem": extension_sem * sigma,
        "angle": mean_angle,
        "angle std": angle_std,
        "angle sem": angle_sem,
        "FU fraction": FU_fraction,
        "UU fraction": UU_fraction,
        "FU mean duration": FU_mean_duration * tau,
        "FU mean duration std": FU_mean_duration_std * tau,
        "FU mean duration sem": FU_mean_duration_se * tau,
        "UU mean duration": UU_mean_duration * tau,
        "UU mean duration std": UU_mean_duration_std * tau,
        "UU mean duration sem": UU_mean_duration_se * tau,
        "FU equilibrium constant": K_eq_FU,
        "UU equilibrium constant": K_eq_UU
        }

        if (FU_fraction > 0 and FU_mean_duration == 0) or (UU_fraction > 0 and UU_mean_duration == 0):
            print(f"[WARNING] Fraction > 0 but duration == 0 → Force: {force}, Seed: {seed}")
            print(f"  FU fraction: {FU_fraction}, FU duration: {FU_mean_duration}")
            print(f"  UU fraction: {UU_fraction}, UU duration: {UU_mean_duration}")
        elif (FU_fraction > 0 and np.isnan(FU_mean_duration)) or (UU_fraction > 0 and np.isnan(UU_mean_duration)):
            print(f"[WARNING] Fraction > 0 but duration == NaN → Force: {force}, Seed: {seed}")

        rows.append(row)

    monomer_df = pd.DataFrame(rows)
    monomer_df = computeSeedAveraged(monomer_df)
    monomer_df.sort_values(by='force')

    df1 = data_df.groupby(['force']).agg({
        'lengths': lambda x: np.concatenate(x.to_numpy()),  # shape: (num_groups, 2, N)
        'angle': lambda x: np.concatenate(x.to_numpy())
    })

    summed_lengths = np.array([arr[0, :] + arr[1, :] for arr in df1['lengths']])
    angles = np.array([val for val in df1['angle']])/180

    mean_r = np.mean(summed_lengths, axis=1) / (2)
    mean_t = np.mean(angles, axis=1)

    std_r = np.std(summed_lengths, axis=1) / (2)
    std_t = np.std(angles, axis=1)

    std_sem_r = sem(summed_lengths, axis=1) / (2)
    std_sem_t = sem(angles, axis = 1)

    monomer_df.loc[df1.index, ('extension', 'std')] = std_r
    monomer_df.loc[df1.index, ('extension', 'mean')] = mean_r
    monomer_df.loc[df1.index, ('extension', 'std_sem')] = std_sem_r

    monomer_df.loc[df1.index, ('angle', 'std')] = std_t
    monomer_df.loc[df1.index, ('angle', 'mean')] = mean_t
    monomer_df.loc[df1.index, ('angle', 'std_sem')] = std_sem_t

    os.chdir('..')
    os.chdir('..')
    path = os.path.join(os.getcwd(), r'data files')
    os.chdir(path)
    print(os.getcwd())

    monomer_df.to_csv('data.csv')

    os.chdir('..')
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)

    plt.figure()

    plt.savefig('moleculeFE.png', dpi=600, bbox_inches='tight')

    plt.figure()
    x = np.linspace(0.01, 0.5, 100)*1000
    y = np.sqrt(T_LJ/(2*x))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('spring constant (lj)')
    plt.ylabel('bond length std (m)')
    plt.title('Length std against spring constant at 0N')
    plt.plot(x, y, color='red')
    plt.errorbar(49, std_r[0], \
                yerr=std_sem_r[0], capsize=2, ecolor='black', fmt='.', \
                label=f'bond length ratio std {round_sig(std_r[0])} ± \
                {round_sig(std_sem_r[0])}')
    plt.legend(loc='upper right')
    plt.savefig('blength_std_vs_spring_constant.png', dpi=600)

    plt.figure(3)
    x = np.linspace(0, 100, 200)
    y = np.sqrt(T_LJ/(x))/np.pi
    plt.title('Angle std against spring constant at 0N')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('angular spring constant (lj)')
    plt.ylabel('angle std (rad)')
    plt.plot(x, y, color='red')
    plt.errorbar(2.3, std_t[0], \
                yerr=std_sem_t[0], capsize=2, ecolor='black', fmt='.', \
                label=f'angle std ratio {round_sig(std_t[0])} ± \
                {round_sig(std_sem_t[0])}')
    plt.legend(loc='upper right')
    plt.savefig('bangle_std_vs_spring_constant.png', dpi=600)

    plt.show()

main()