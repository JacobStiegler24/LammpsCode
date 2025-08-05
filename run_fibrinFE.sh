#!/bin/bash


#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer-FE.lam"
seed="54654651 54654653 54654659 54654661 54654685"
outdir="output"

#=======================================


# Run simulations
for seedval in $seed ; do
  outdir="output/ForceExtension/Force-ExtensionSeed${seedval}"
  mkdir -p "$outdir"
  ./lmp_mpi \
    -var seed "$seedval" \
    -var outdir "$outdir" \
    < "$lmpinput"

done
