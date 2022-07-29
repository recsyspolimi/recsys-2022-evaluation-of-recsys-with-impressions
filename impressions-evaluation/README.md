# Package: Impressions Evaluation

This package is the only entry-point to execute the experiments of the paper 
"Towards the Evaluation of Recommender Systems with Impressions". The environment and dependencies is automatically 
managed by `poetry`. If you wish to run any experiment, then go through the installation instructions, starting at 
the [Starting point section](#starting-point). 

## Package organization

### Starting point

The only starting point to execute the experiments is the script [main.py](main.py), which executes experiments
and prints results based on console arguments. 

Please, refer to the [Installation](#installation) and then the
[Experiments](#experiments-source-code) sections to know how to install the dependencies of the project in your OS and
how to run our experiments, respectively.

### Dataset Processing source code.
Three different files contain the code to load, process, compute features, and save to disk the datasets. 
- For the ContentWise Impressions dataset, see [ContentWiseImpressionsReader.py](ContentWiseImpressionsReader.py)
- For the MIND-SMALL dataset, see [MINDReader.py](MINDReader.py)
- For the FINN.no Slates dataset, see [FINNNoReader.py](FINNNoReader.py)

### Experiments source code

The [main.py](main.py) script is the main orchestrator to run experiments. Which experiments are run are controlled by 
the following console flags.
```python
class ConsoleArguments:
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if 
    possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """If the flag is included, then the script tunes the hyper-parameters of the base recommenders, e.g., ItemKNN, 
    UserKNN, SLIM ElasticNet."""

    include_folded: bool = False
    """If the flag is included, then the script folds the tuned matrix-factorization base recommenders. If the 
    recommenders are not previously tuned, then this flag fails."""

    include_impressions_time_aware: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of time-aware impressions recommenders: 
    Last Impressions, Recency, and Frequency & Recency. These recommenders do not need base recommenders to be tuned."""

    include_impressions_reranking: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of re-ranking impressions recommenders: 
    Cycling and Impressions Discounting. These recommenders need base recommenders to be tuned, if they aren't then 
    the method fails."""

    include_ablation_impressions_reranking: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of re-ranking impressions recommenders: 
    Impressions Discounting with only impressions frequency. These recommenders need base recommenders to be tuned, 
    if they aren't then the method fails."""

    include_impressions_profile: bool = False
    """If the flag is included, then the script tunes the hyper-parameter of impressions as user profiles recommenders. 
    These recommenders need similarity-based recommenders to be tuned, if they aren't then the method fails."""

    print_evaluation_results: bool = False
    """Export to CSV and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of 
    all tuned recommenders."""
```

## Installation

Note that this repository requires `Python 3.9`, `poetry`, `Cython`, the
[recsys framework](../RecSysFramework_public/README.md), 
and the [recsys framework extensions](../recsys-framework-extensions/). 
We tested our installation against [Linux](#linux-installation) and [macOS](#macos-installation). 
Currently, the package does not support installations on Windows.

### Download Datasets

 To run the experiments, you must download each dataset separately and place it in the corresponding folder for each 
 dataset. The following map illustrates where to put the uncompressed files of each dataset. The scripts will do 
 `a best effort attempt` to download the datasets, however, it may not be reliable due to network conditions. 
```
RecSys-2022-Towards-the-Evaluation-of-Recommender-Systems-with-Impressions/
  |----> impressions-evaluation/
  |      |
  |      |---->data/
  |      |      |---->ContentWiseImpressions/
  |      |      |      |---->original/
  |      |      |      |      |  <Place dataset here> 
  |      |      |---->FINN-NO-SLATE/
  |      |      |      |---->original/
  |      |      |      |      |  <Place dataset here> 
  |      |      |---->MIND-SMALL/
  |      |      |      |---->original/
  |      |      |      |      |  <Place dataset here> 
  |      | ...
  |----> RecSysFramework_public/
  |      | ...
  |----> recsys-framework-extensions/
  |      | ...
  ...
```

Now you'll need to follow the installation procedures of the target OS:

- [Linux](#linux-installation)
- [macOS](#macos-installation).

Be aware that during the compilation you may see some warnings. The installation procedures for your OS will guide you
through all the steps needed to execute our experiments.

### Linux Installation

- Enter the `impressions-dataset` folder:
  ```bash
  cd impressions-evaluation/
  ```
- Install dependencies for `pyenv`, `poetry`, and the repo source code (this includes a C/C++ compiler).
  ```bash
  sudo apt-get update -y; sudo apt-get install gcc make python3-dev gifsicle build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
  ```
- `Python 3.9.7` using [pyenv](https://github.com/pyenv/pyenv#installation)
  ```bash
  curl https://pyenv.run | bash
  ```
    - Remember to include `pyenv` in your shell
      [Section 2: Configure your shell's environment for Pyenv](https://github.com/pyenv/pyenv#basic-github-checkout).
    - Reload your shell (simple: quit and open again).
- `Poetry` using `pyenv`
   ```bash
   pyenv install 3.9.7
   pyenv local 3.9.7
   curl -sSL https://install.python-poetry.org | python3 -
   ```
    - Ensure to add `export PATH="/home/<your user>/.local/bin:$PATH"` to your bash profile (e.g., `~/.bashrc`
      , `~/.bash_profile`, etc)
- Download dependencies using `poetry`
  ```bash
  poetry install
  ``` 

### macOS Installation

- Enter the `impressions-dataset` folder:
  ```bash
  cd impressions-dataset/
  ```
- `Command Line Tools for Xcode` from the [Apple's Developer website](https://developer.apple.com/download/more/?=xcode)
  . Required to have a `C` compiler installed in your Mac. You'll need a free Apple ID to access these resources.
  ```bash
  xcode-select --install
  ```
- `Homebrew` from [this page](https://brew.sh)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew update
   brew install openssl readline sqlite3 xz zlib hdf5 c-blosc
   ```
- `Python 3.9.7`
    - Using [pyenv](https://github.com/pyenv/pyenv#installation)
      ```bash
      curl https://pyenv.run | bash
      ```
- `Poetry` using `pyenv`
    ```bash
    pyenv install 3.9.7
    pyenv local 3.9.7
    curl -sSL https://install.python-poetry.org | python3 -
    ```
- Download dependencies using `poetry`
  ```bash
  poetry install
  ```

## Experiments

You can re-run all the experiments that by following the instructions in the
[hyper-parameter tuning](#hyper-parameter-tuning) sections.

### Parallelize execution

This repository uses [dask](https://dask.org) to parallelize the experiments using processes. We used AWS 
instances to run all the experiments. Depending on the number of available cores on your machine, you will need to 
adapt the default number of processes. 

By default, this repository uses `2` different processes, you can however, change this
default by changing the `num_workers` key inside the [pyproject.toml](pyproject.toml) file. 
For example, if you want to use 4 processes, then change the key to `num_workers = 4`. 
If you want to disable parallelism, then set the key to `num_workers = 1`.


### Hyper-parameter tuning.

The main script [main.py](main.py) executes and prints the results of the hyper-parameter tuning of different
recommenders. By default, the main script without console arguments just exists successfully.

Replicating our experiments is as easy as passing all console flags. The framework will wait for results as needed by 
the recommenders.

```bash
poetry run python main.py \
 --create_datasets \
 --include_baselines \
 --include_folded \
 --include_impressions_time_aware \
 --include_impressions_reranking \
 --include_ablation_impressions_reranking \
 --include_impressions_profile \
 --print_evaluation_results
```
