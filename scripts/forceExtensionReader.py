print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import re as re
print('imported')
# Constants
timestep = 0.002 # lj
nevery = 5 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5 # nm
epsilon = 225 # nm*pN
mass = 2.12 # E^-22 kg
tau = sigma*np.sqrt(mass/epsilon)

def readfile():
    # Change directory to the project folder
    os.chdir(r'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer\output\forceExtension')
    print(os.getcwd())

    force_lst = [] 

    folder_lst = os.listdir()
    
    # https://docs.python.org/3/library/re.html
    # https://docs.python.org/3/howto/regex.html#regex-howto
    for folder in folder_lst:
        match = re.search(r'Force(\d+\.\d+)?', folder)
        if match and (match.group(0) not in force_lst):
            force_lst.append(match.group(0))

    force_lst.sort()
    folder_lst.sort()

    data_dict = {} # hierarchy: force -> seed -> types, lengths
    counter = 0
    for folder in folder_lst:
        os.chdir(rf'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer\output\forceExtension\{folder}')
        btype = [[],[]]
        blen = [[],[]]
        with open('bondlengths.dump') as f:
            isItem = False
            counter2 = 0
            for line in f:
                if isItem:
                    btype[counter2].append(line[0])
                    blen[counter2].append(line[2::].strip(' \n'))
                    counter2 += 1
                if counter2 == 2:
                    isItem = False
                    counter2 = 0
                if 'ENTRIES c_btype c_bondlen' in line:
                    isItem = True

        btype_arr = np.array(btype).astype(int)
        blen_arr = np.array(blen).astype(np.float64)
        
        for force in force_lst:
            if force in folder:
                found_force = float(force.strip('Force'))
                if found_force not in data_dict:
                    data_dict[found_force] = {}
                    counter = 0
        
        data_dict[found_force][counter] = {
            'types': btype_arr,
            'lengths': blen_arr
        }
        counter += 1

    return data_dict

def lengthStats(blen_arr):
    mean_extension = np.mean(blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0])
    print(f'mean seed extension: {mean_extension}')
    return mean_extension


data_dict = readfile()

analysis_dict = {} # hierarchy: force -> mean extension

force_lst = list(data_dict.keys())
force_lst.sort()
seedN_lst = list(data_dict[force_lst[0]].keys())

extension_lst = []
for force in force_lst:
    mean_seed_extension_lst = []
    for seed in seedN_lst:
        mean_seed_extension_lst.append(lengthStats(data_dict[force][seed]['lengths'])*sigma)
    extension_lst.append(np.mean(mean_seed_extension_lst))

for index in range(len(force_lst)):
    force_lst[index] = float(force_lst[index])*10*epsilon/sigma

os.chdir(r'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer')

plt.figure(1)

plt.scatter(extension_lst, force_lst, c = 'r')
plt.plot(extension_lst, force_lst)

plt.xlabel('extension (nm)')
plt.ylabel('force (pN)')   
plt.title('Molecule Force-Extension')
plt.legend()

plt.savefig('figures/moleculeFE.png', dpi=600, bbox_inches='tight')
plt.show()