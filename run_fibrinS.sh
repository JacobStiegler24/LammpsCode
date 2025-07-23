#!/bin/bash


#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer.lam"
stretch_force=0.0 # must be a decimal
seed="54654651 54654653 54654659 54654661 54654685"
outdir="output"

#=======================================


# Run simulations
for seedval in $seed ; do
  outdir="output/single/Force${stretch_force}_Seed${seedval}"
  mkdir -p "$outdir"
  ./lmp_mpi \
    -var stretch_force "$stretch_force" \
    -var seed "$seedval" \
    -var outdir "$outdir" \
    < "$lmpinput"

done
