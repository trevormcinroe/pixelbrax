# pixelbrax

## Installation
```
conda create -n pixelbrax python=3.9
conda activate pixelbrax
pip install -U "jax[cuda12]==0.4.30"
pip install flax
pip install jaxtyping==0.2.0
pip install tdqm
pip install mujoco-mjx==3.2.6
pip install mujoco==3.2.6
```

After cloning the repo, `cd pixelbrax`, then install via
```bash
pip install .
```

If you still have missing packages, or you encounter mujoco/mjx related errors, try the following. Be mindful of the order of `mujoco-mjx` and `mujoco` installations. If not respected, `mujoco-mjx` may override the correct `mujoco` version.
```
pip install brax
pip uninstall brax
pip install mujoco-mjx==3.2.6
pip install mujoco==3.2.6
```

Download the DAVIS dataset:

```
mkdir datasets
cd datasets
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
```

To create an environment, run
```python
envs, _, _ = pixelbrax.make(
    backend="spring",
    env_name="halfcheetah",
    n_envs=100,
    seed=0,
    hw=84,
    distractor=None,
    video_path="datasets/DAVIS",
    video_set="train",
    return_float32=False,
)
```

