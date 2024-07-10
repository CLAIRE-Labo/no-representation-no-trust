ls outputs/release/solve/atari-ppo/baselines | xargs printf -- '"outputs/release/solve/atari-ppo/baselines/%s",\n' > ../dev/tmp.tmp
for control in algo all optimizer regularize regularize-all-layers shared-trunk; do
    ls outputs/release/solve/atari-ppo/control/$control | xargs printf -- '"outputs/release/solve/atari-ppo/control/'${control}'/%s",\n' >> ../dev/tmp.tmp
done
