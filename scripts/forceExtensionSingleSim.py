print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
print('imported')

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
    path = os.path.join(os.getcwd(), r'projects/Fibrin-Monomer/output/ForceExtension')
    
    # path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\ForceExtension'
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
        
        force_lst = []
        with open('netforce.out') as f:
            next(f)  # skip first line
            force_arr = np.array([float(line.strip()) for line in f], dtype=float)
        row = {
            "seed": seed,
            "force": force_arr,
            "lengths": np.array(blen_lst, dtype=np.float64),
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
    print(os.getcwd())
    os.chdir('..')
    print(os.getcwd())
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    mforce_arr_series = data_df['force']
    blen_arr_series = data_df['lengths']
    btype_arr_series = data_df['types']

    sumrange = 1000
    plt.figure(1)
    sc = None
    for i in range(blen_arr_series.shape[0]):
        btype_arr = btype_arr_series[i]
        blen_arr = blen_arr_series[i]
        mforce_arr = mforce_arr_series[i]

        cumsum_force = np.cumsum(mforce_arr)
        cumsum_force[sumrange:] = cumsum_force[sumrange:] - cumsum_force[:-sumrange]
        moving_avg_force = cumsum_force[sumrange-1:] / sumrange

        mtype_arr = btype_arr[0] + btype_arr[1]
        bextension_arr = (blen_arr[0]+blen_arr[1]-blen_arr[0,0]-blen_arr[1,0])

        cumsum_extension = np.cumsum(bextension_arr)
        cumsum_extension[sumrange:] = cumsum_extension[sumrange:] - cumsum_extension[:-sumrange]
        moving_avg_extension = cumsum_extension[sumrange-1:] / sumrange

        cumsum_type = np.cumsum(mtype_arr)
        cumsum_type[sumrange:] = cumsum_type[sumrange:] - cumsum_type[:-sumrange]
        moving_avg_type = cumsum_type[sumrange-1:] / sumrange

        plt.plot(moving_avg_extension, moving_avg_force, linewidth=0.7)
        plt.scatter(moving_avg_extension, moving_avg_force, s=2.5, alpha=0.7)

        sc = plt.scatter(
            moving_avg_extension,
            moving_avg_force,
            s=2.5,
            alpha=0.7,
            c=moving_avg_type,
            cmap='viridis',
            label=f'Seed {data_df["seed"][i]}'
        )

    plt.xlabel('Extension (m)')
    plt.ylabel('Force (N)')
    plt.title('Molecule Force/Extension')

    # Add colourbar for type
    cbar = plt.colorbar(sc)
    cbar.set_label('Type')

    plt.xlabel('Extension (m)')
    plt.ylabel('Force (N)')
    plt.title(f'Molecule Force/Extension')

    plt.savefig(f'MoleculeMoveFE.png', dpi=600, bbox_inches='tight')

    plt.show()

def main():
    data_df = readfile()
    plotgraphs(data_df)

main()