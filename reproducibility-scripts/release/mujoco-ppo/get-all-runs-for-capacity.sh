ls outputs/release/solve/mujoco-ppo/baselines | xargs printf -- '"outputs/release/solve/mujoco-ppo/baselines/%s",\n' > ../dev/tmp.tmp
for control in algo optimizer regularize shared-trunk; do
    ls outputs/release/solve/mujoco-ppo/control/$control | xargs printf -- '"outputs/release/solve/mujoco-ppo/control/'${control}'/%s",\n' >> ../dev/tmp.tmp
done
