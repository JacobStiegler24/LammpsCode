print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
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


def readfile():
    # Change directory to the project folder
    path = os.path.join(os.getcwd(), r'output')
    os.chdir(path)
    print(os.getcwd())
    

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        
        seed_match = re.search(r'Force-ExtensionSeed(\d+)', folder)
        

        if not (seed_match):
            continue

        print(seed_match)

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
        
        force_arr = []
        with open('force.dump') as f:
            isItem = False
            counter2 = 0
            netForce = 0
            for line in f:
                if isItem:
                    netForce += float(line)
                    counter2 += 1
                if counter2 == 1:
                    isItem = False
                    force_arr.append(netForce)
                    netForce = 0
                    counter2 = 0
                if 'ITEM: ATOMS fz' in line:
                    isItem = True

        row = {
            "seed": seed,
            "force": np.array(force_arr, dtype=np.float64)*F,
            "lengths": np.array(blen_lst, dtype=np.float64)*sigma,
            "types": np.array(btype_lst, dtype=int)
        }
        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1


    df = pd.DataFrame(rows)

    return df

def plotgraphs(data_df):
    # Plotting the b lengths over time
    os.chdir(r'..') 
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    mforce_arr_series = data_df['force']
    blen_arr_series = data_df['lengths']
    btype_arr_series = data_df['types']

    plt.figure(3)
    for i in range(blen_arr_series.shape[0]):
        btype_arr = btype_arr_series[i]
        blen_arr = blen_arr_series[i]
        mforce_arr = mforce_arr_series[i]
        mtype_arr = btype_arr[0] + btype_arr[1]
        bextension_arr = (blen_arr[0]+blen_arr[1]-blen_arr[0,0]-blen_arr[1,0]) * sigma
        plt.plot(bextension_arr, mforce_arr, linewidth=0.7)
        plt.scatter(bextension_arr, mforce_arr, s=0.5, alpha=0.5, c=mtype_arr)

    plt.xlabel('Extension (m)')
    plt.ylabel('Force (N)')
    plt.title(f'Molecule Force/Extension')

    plt.savefig(f'MoleculeMoveFE.png', dpi=600, bbox_inches='tight')

    plt.show()

def main():
    data_df = readfile()
    plotgraphs(data_df)

main()