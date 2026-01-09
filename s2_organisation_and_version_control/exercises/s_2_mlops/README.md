# exercises_s2

Exercises for s2 M6: code structure

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## How to run
Run all commands from repository root

### 1. Preprocess data

Loads raw corrupt MNIST data, concatenates train and test images into a single
dataset, normalizes images to mean 0 and standard deviation 1, and saves the
processed tensors.

```bash
uv run src/exercises_s2/data.py preprocess data/raw/corruptmnist_v1  data/processed
```
outputs:
- data/processed/images.pt
- data/processed/labels.pt

### 2. Create dataset splits
Creates reproducible train/validation/test splits and stores the indices.

```bash
uv run src/exercises_s2/splits.py
```

output:
- data/splits.pt

### 3. Train the model
Trains the CNN using the training split, evaluates on the validation split each
epoch, applies early stopping, and saves training statistics.

```bash
uv run src/exercises_s2/train.py
```

outputs:
- Model checkpoints in models/ (e.g. best_model.pth)
- Training plots in reports/figures/


### 4. Evaluate the model
Evaluates a trained model checkpoint on the test split and prints metrics.
```bash
uv run src/exercises_s2/evaluate.py models/best_model.pth
```
Printed output:
- Test accuracy
- Test F1 score (macro)


### 5. Visualize embeddings
Extracts intermediate feature representations from the trained model and
visualizes them in 2D using t-SNE.
```bash
uv run src/exercises_s2/visualize.py models/best_model.pth
```

Output
- reports/figures/embeddings.png

Notes
- Raw data in data/raw is not tracked by Git.
- Dataset splits are saved once and reused to ensure reproducibility.
- All paths assume the directory structure shown above.
- The visualization uses features from the network just before the final classification layer.