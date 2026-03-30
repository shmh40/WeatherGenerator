## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current develop.
3. **Read the in-scope files**: The repo is large. Read these directories for full context:
   - `src/weathergen/`— model code that you can modify. Model architecture, optimizer, training loop.
   - `config/` — the configuration files that you can modify. 
5. **Check or initialize results.tsv**: Check if `results.tsv` exists. If not, create it with just the header row. The baseline will be recorded after the first run. If it already exists, just keep experimenting where you left off -- you can inspect branches to see what has been tried.
6. **Confirm and go**: Confirm that you have an understanding and you are ready to go.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a 1 node, with 4 GPUs. The training script runs will run for a **fixed** 64 mini epochs with a fixed number of samples per mini epoch. Do not change this. You launch it simply as: `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/EXPERIMENT_CONFIG.yml`, where `EXPERIMENT_CONFIG.yml` is a config file you create for this experiment (you can copy from previous ones and modify). Note this command also sets off a cleanup script, which is not relevant. You do not need to read the WeatherGenerator-private repository, or the launch-slurm.py script, it simply sets off a slurm job beginning training.

**What you CAN do:**
- Modify files in `src/weathergen/` — this is the only directory you edit. All files in here are fair game: model architecture, optimizer, hyperparameters, training loop and so on. 
- You can also modify the config files in `config/`, including `config/streams/` to change hyperparameters and other settings.

**What you CANNOT do:**
- Modify `../WeatherGenerator-private/`. This is **read-only**. It contains data loading and training constants (time budget, number of nodes, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the training task, which is defined in the config here:

```yaml
  losses : {
    "physical": {
        type: LossPhysical,
        loss_fcts: { "mse": { }, },
        },
    }
  
  model_input: {
    "forecasting" : {
      # masking strategy: "random", "healpix", "forecast"
      masking_strategy: "forecast",
      },
    }

  forecast :
      time_step: 06:00:00
      offset: 1
      num_steps: 2
      policy: "fixed"
```
- Modify the number of mini_epochs, number of samples, or the training start_date and end_date, or time_window_step or time_window_len, all of which are defined in the config as follows:

```yaml
  num_mini_epochs: 64
  samples_per_mini_epoch: 4096
  shuffle: True

  start_date: 1979-01-01T00:00
  end_date: 2022-12-31T00:00

  time_window_step: 06:00:00
  time_window_len: 06:00:00
```

- Modify the validation_config component of the config.

**The goal is simple: get the lowest validation loss.** Since the number of mini epochs is fixed, you don't need to worry about training time — it's always 64 mini epochs. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful validation loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 validation loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 validation loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is, using `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/default_config.yml`, without any modifications.

## Output format

The slurm script will generate a random 8 character RUN_ID, which will be printed in the terminal. Once the slurm script has submitted its job and is running, it will save results to a log file, located here: /hpcperm/ecm8347/work/wg_autoresearch/WeatherGenerator/results/RUN_ID/RUN_ID_train_metrics.json, which has the following format, where the key metric is the latest loss.LossPhysical.ERA5.mse.loss_avg during the "val" stage:

```
{"weathergen.timestamp": 1764691348605, "weathergen.time": 20251202160228, "stage": "train", "num_samples": 32.0, "loss_avg_mean": 0.3831610083580017, ... }
{"weathergen.timestamp": 1764691390000, "weathergen.time": 20251202160310, "stage": "val", "num_samples": 16.0, "loss.LossPhysical.ERA5.mse.loss_avg": 1.0551681679337181, ... }
```

Note that the script is configured to always stop after 64 mini epochs. You should extract the key metric from the log file, which is the latest loss.LossPhysical.ERA5.mse.loss_avg at the "val" stage, and evaluate whether it's an improvement over the baseline. You can monitor jobs using: `squeue | grep weathergen`, since `squeue -u $USER` will not be available to you.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	loss.LossPhysical.ERA5.mse.loss_avg	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. loss.LossPhysical.ERA5.mse.loss_avg achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	loss.LossPhysical.ERA5.mse.loss_avg	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar30` or `autoresearch/mar30-learning-rate`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `src/weathergen/` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/EXPERIMENT_CONFIG.yml` (make sure to specify the correct config file for this experiment, which should be in the same branch and should have a unique name so you can keep track of it). Wait until the slurm job is submitted, which you can confirm by inspecting the terminal output after this command.
5. While you wait for this experiment to run, you can start thinking about the next experiment and preparing the code changes for it, but you MUST commit those changes to a NEW BRANCH (e.g. `autoresearch/mar30-exp2`) before again launching the slurm job. The launch-slurm copies only what is currently committed to be run in the slurm job. This is very important. This way you can have multiple experiments running in parallel, but keep the code changes for each experiment organized in separate branches.
6. When a given slurm job is finished (you can check with `squeue -u $USER`), read out the results from the RUN_ID_train_metrics.json file, and extract the key metric loss.LossPhysical.ERA5.mse.loss_avg at the "val" stage, and compare to baseline. If it's an improvement, keep it. If it's not, discard.
6. If the output is empty, the run crashed. Inspect /hpcperm/ecm8347/work/wg_autoresearch/WeatherGenerator/output/output_RUNID_SLURMID.txt to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If loss.LossPhysical.ERA5.mse.loss_avg improved (lower), you "advance" the branch, keeping the git commit
9. If loss.LossPhysical.ERA5.mse.loss_avg is equal or worse, you git reset back to where you started
10. NOTE: since you can run multiple experiments in parallel, you should be VERY CAREFUL to commit each change to separate branches (e.g. `autoresearch/mar30-exp1`, `autoresearch/mar30-exp2`, etc) before running the experiment with WeatherGenerator in THAT BRANCH, and then only switching to a new idea (and hence new branch) once the launch-slurm script has completed successfully. Then you should merge a successful branch back to the main experiment branch (e.g. `develop`) only if it's an improvement. This way you can keep the history clean and avoid confusion.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~24 hours total (+ a few seconds for startup and eval overhead). If a run exceeds 25 hours, kill it.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, end of.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~24 hours, and you can run 8 experiments at one time, then you can run approx 8/day, for a total of about 56 over the duration of the average week.