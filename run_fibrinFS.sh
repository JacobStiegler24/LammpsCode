#!/bin/bash


#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer.lam"
stretch_force=$(seq 0 2 20 | xargs -n1 printf "%.1f\n") # must be decimal values. 
#| xargs -n1 printf "%.1f\n" from chatgpt to make forces a decimal for all increments.
seed="54654651 54654653 54654659 54654661 54654685"
outdir="output"

#=======================================

 
#=======================================
rm -rf output
#=======================================


# Run simulations
for force in $stretch_force ; do
  for seedval in $seed ; do
    outdir="output/Force${force}_Seed${seedval}"
    mkdir -p "$outdir"
    ./lmp_mpi \
      -var stretch_force "$force" \
      -var seed "$seedval" \
      -var outdir "$outdir" \
      < "$lmpinput"

  done
done
