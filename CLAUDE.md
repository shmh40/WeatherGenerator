## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr27`). The branch `autoresearch/performance/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/performance/<tag>` from current `develop-apr27`. **Note** `develop-apr27` is the main branch for these experiments, **not** `develop`.
3. **Read the in-scope files**: The repo is large. Read these directories for full context:
   - `src/weathergen/`— model code that you can modify. Model architecture, optimizer, training loop.
   - `config/` — the configuration files that you mostly cannot modify. 
5. **Check or initialize results.tsv**: Check if `results.tsv` exists. If not, create it with just the header row. The baseline will be recorded after the first run. If it already exists, just keep experimenting where you left off -- you can inspect branches to see what has been tried.
6. **Confirm and go**: Confirm that you have an understanding and you are ready to go.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a 1 node, with 4 GPUs. The training script runs will run for a **fixed** 1 mini epoch with a fixed number of samples per mini epoch. Do not change this. You launch it simply as: `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/EXPERIMENT_CONFIG.yml`, where `EXPERIMENT_CONFIG.yml` is a config file you create for this experiment (you can copy from previous ones and modify). Note this command also sets off a cleanup script, which is not relevant. You do not need to read the WeatherGenerator-private repository, or the launch-slurm.py script, it simply sets off a slurm job beginning training.

**What you CAN do:**
- Modify files in `src/weathergen/` — this is the only directory you edit. All files in here are fair game: model architecture, optimizer, hyperparameters, training loop and so on. 
- You cannot modify most of the config files. There are a small number of parameters you can modify in default_config.yml:

```
with_mixed_precision: True
with_flash_attention: True
compile_model: False
with_fsdp: True
attention_dtype: bf16
mixed_precision_dtype: bf16
```

and 

```
# parameters for data loading
data_loading :

  num_workers: 12
  rng_seed: ???
  repeat_data_in_mini_epoch : False

  # pin GPU memory for faster transfer; it is possible that enabling memory_pinning with 
  # FSDP2 + DINOv2 can cause the job to hang and trigger a PyTorch timeout error.
  # If this happens, you can disable the flag, but performance will drop on GH200.
  memory_pinning: True
```

 Everything else in this config **must remain the same**.

**What you CANNOT do:**
- Modify `../WeatherGenerator-private/`. This is **read-only**. It contains data loading and training constants (time budget, number of nodes, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the training task, number of samples, or mini epochs, which is defined in the default_config.yml.

**The goal is simple: maximise the throughput of training.** Since the number of mini epochs is fixed, and the number of samples per mini-epoch, you only need to worry about the samples per second of throughput. The operations of the code **must remain the same**, but the samples per second must improve. The only constraint is that the code runs without crashing, and that we do not or will not run out of memory.

**The loss** is a hard constraint. Some very minor increase is acceptable, but this must be due to seed or other noise. It should not be degraded significantly compared to the baseline.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing some code and getting equal or better throughput while keeping the functionality the same is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 samples per second improvement that adds 30 lines of hacky code? Probably not worth it. A 0.001 samples per second improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is, using `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/default_config.yml`, without any modifications.

## Output format

The slurm script will generate a random 8 character RUN_ID, which will be printed in the terminal. Once the slurm script has submitted its job and is running, it will save results to an output log file, located here: /hpcperm/ecm8347/work/wg_autoresearch_perf/WeatherGenerator/output/output_RUNID_SLURMID.txt, which has the following format, where the key metric is s/sec, and the loss:

```
...
0: 000 : 00010/00512 : 000010 : loss = 3.5386E-02 (lr=6.08E-07, s/sec=0.161)
0: 	ERA5 : 3.5386E-02 	
0: 
0: 000 : 00020/00512 : 000020 : loss = 3.4524E-02 (lr=1.29E-06, s/sec=0.360)
0: 	ERA5 : 3.4524E-02 	
0: 
0: 000 : 00030/00512 : 000030 : loss = 3.2183E-02 (lr=2.38E-06, s/sec=0.358)
0: 	ERA5 : 3.2183E-02 	
0: 
0: 000 : 00040/00512 : 000040 : loss = 3.1569E-02 (lr=3.86E-06, s/sec=0.366)
0: 	ERA5 : 3.1569E-02 	
0: 
0: 000 : 00050/00512 : 000050 : loss = 2.9302E-02 (lr=5.72E-06, s/sec=0.361)
0: 	ERA5 : 2.9302E-02 
...	
```

Note that the script is configured to always stop after 1 mini epoch. You should extract the key metrics from the log file, which is the s/sec *averaged* across all steps, and evaluate whether it's an improvement over the baseline. At the same time, you must validate that the final validation loss is not more that 10% higher than the baseline. You can monitor jobs using: `squeue | grep weathergen`, since `squeue -u $USER` will not be available to you.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	average_ssec loss_degradation memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. average_ssec — s/sec, averaged across all steps during training. Use 0.000000 for crashes
3. loss_degradation - was the final validation loss more than 10% higher than the baseline
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	average_ssec    loss_degradation	memory_gb	status	description
a1b2c3d	0.324	       no     44.0	keep	baseline
b2c3d4e	0.312	       no     44.2	keep	parallelise per variable in sampling
c3d4e5f	0.432	       yes    44.0	discard	vectorise masking
d4e5f6g	0.000000	   0.0     0.0	crash	change GPU all gather
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/performance/apr27` or `autoresearch/performance/apr27-vectorise-mask`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `src/weathergen/` with an idea by directly editing the code.
3. git commit
4. Run the experiment: `../WeatherGenerator-private/hpc/launch-slurm.py --base-config config/EXPERIMENT_CONFIG.yml` (make sure to specify the correct config file for this experiment (usually default_config.yml), which should be in the same branch and should have a unique name so you can keep track of it). Wait until the slurm job is submitted, which you can confirm by inspecting the terminal output after this command.
5. While you wait for this experiment to run, you can start thinking about the next change and preparing the code changes for it, but you MUST commit those changes to a NEW BRANCH (e.g. `autoresearch/performance/apr27-exp2`) before again launching the slurm job. The launch-slurm copies only what is currently committed to be run in the slurm job. This is very important. This way you can have multiple experiments running in parallel, but keep the code changes for each experiment organized in separate branches.
6. When a given slurm job is finished (you can check with `squeue -u $USER`), read out the results from the output_RUNID_SLURMID.txt file, and extract the key metric s/sec, averaged across all steps, and compare to baseline. If it's an improvement, keep it. If it's not, discard.
6. If the output is empty, the run crashed. Inspect /hpcperm/ecm8347/work/wg_autoresearch_perf/WeatherGenerator/output/output_RUNID_SLURMID.txt to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If s/sec improved (higher), you "advance" the branch, keeping the git commit
9. If s/sec is equal or worse, you git reset back to where you started
10. NOTE: since you can run multiple experiments in parallel, you should be VERY CAREFUL to commit each change to separate branches (e.g. `autoresearch/performance/apr27-exp1`, `autoresearch/performance/apr27-exp2`, etc) before running the experiment with WeatherGenerator in THAT BRANCH, and then only switching to a new idea (and hence new branch) once the launch-slurm script has completed successfully. Then you should merge a successful branch back to the main experiment branch (`develop-27apr`) only if it's an improvement. This way you can keep the history clean and avoid confusion.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take less than 1 hour total (+ a few seconds for startup and eval overhead). 

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, end of.

As an example use case, a user might leave you running while they sleep. If each experiment takes you less than an hour, and you can run 8 experiments at one time, then you can run approx 192/day, for a total of about 64 over the duration of a human's sleep.