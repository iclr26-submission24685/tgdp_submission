# Temperature Guided Diffusion Planner

A PyTorch implementation of the temperature guided diffusion planner algorithm described in [Practical Diffusion Planning via Temperature-Guided Reward Conditioning](https://openreview.net/pdf?id=LOvutpJlgL).

---

## Installation

### Setup MuJoCo for D4RL environments

```bash
# Create MuJoCo dir.
MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR

# Download and unpack MuJoCo 2.1.0.
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz
tar -xzf $MUJOCO_DIR/mujoco210.tar.gz -C $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco210.tar.gz
```

### Install dependencies

```bash
# Clone the repository
git clone https://github.com/iclr26-submission24685/tgdp_submission.git

# Create the conda environment from the environment.yml.
conda env create -f environment.yml

# Activate the conda environment.
conda activate tgdp

# Install pip dependencies. 
pip install --ignore-requires-python -r requirements.txt

# Set environment variables.
conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda env config vars set MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210

# Install tgdp repo.
pip install -e .
```

---

## Running Models

You can run training, evaluation, and hyperparameter optimization for implemented models using the scripts in the `scripts/` directory. Each script corresponds to a different model and can be configured via command-line arguments or configuration files.

Implemented environments include (not all models support all environments):
- Locomotion: `[halfcheetah/hopper/walker2d]-[medium/medium-replay/medium-expert]-v2`.
- Maze2D: `maze2d-[umaze/medium/large]-v1`.
- Kitchen: `kitchen-[partial/mixed]-v0`.

**Training**

```bash
python scripts/<subfolder>/<model_script>.py --mode train --env <env_name>
```

**Evaluation**

```bash
python scripts/<subfolder>/<model_script>.py --mode test --env <env_name>
```

**Hyperparameter Optimization**

```bash
python scripts/<subfolder>/<model_script>.py --mode optim --env <env_name>
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, please open an issue on GitHub.
