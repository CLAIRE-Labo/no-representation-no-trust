N_PARALLEL=$1

for _ in $(seq "$N_PARALLEL"); do
  "${@:2}" &
done

wait

# To use with runai do:
# -- zsh reproducibility-scripts/utils/run-in-parallel.sh <N_PARALLEL> <COMMAND>
