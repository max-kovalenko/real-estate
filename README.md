# real_estate

MLOps course study project

## Setting up the project

```bash
# Installing MiniConda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Creating a virtual environment
conda env create --prefix ./env -f environment.yml

# To remove the long prefix in a shell prompt
conda config --set env_prompt '({name})'

# Activating env inside the project directory
conda activate ./env/

# Turning off poetry venv autocreation locally
poetry config virtualenvs.create false --local

# Initializing poetry
poetry init

# Installing packages
poetry install
```