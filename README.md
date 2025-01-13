# pixelbrax

After cloning the repo, `cd pixelbrax`, then install via
```bash
pip install .
```

You will also need to download our custom version of Brax from: `https://github.com/trevormcinroe/brax_for_pixels`.
The easiest way to connect brax with your training code is to add its path to your `.bashrc` file like: `export PYTHONPATH="/path/to/brax_for_pixels"`

To create an environment, run
```python
envs, _, _ = pixelbrax.make(
    backend="spring",
    env_name="halfcheetah",
    n_envs=100,
    seed=0,
    hw=84,
    distractor=None,
    video_path=$YOUR_DAVIS_PATH,
    video_set="train",
    return_float32=False,
)
```

For the video distractors, you will need to download the 480p "TrainVal" Semi-supervised dataset from: `https://davischallenge.org/davis2017/code.html`, and then set the `video_path` arg to the extracted folder on your machine.

