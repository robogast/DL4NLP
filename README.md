# DL4NLP

# Install
```bash
conda env create -f environment.yml
conda activate dl4nlp
python setup.py develop
```

# Run (on lisa)
(from root) `sbatch slurm-jobs/train_mat.job`

# Run (locally)
(from root) `python train.py <args>`
run `python train.py --help` to see all possible arguments.
See `train_mat.job` for recommended args.