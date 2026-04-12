# Rare-class area estimation

A command-line tool for adaptive stratified sampling to estimate the true area of a rare class in a reference map (e.g. a land-cover or satellite image classification). Implements **Algorithm 1** from:

> Oles V, Boschetti L, Roy DP, Dykhovychnyi O, Tubiello F, Mollicone D "*Accounting for needles amongst the earth observation haystacks: accurate area estimation of rare classes*"

The algorithm accounts for map misclassification by iteratively sampling units from two strata, collecting human-verified misclassification counts, and refining a Bayesian area estimate until a target precision is achieved.

## How it works

Given a map that classifies every unit (e.g. pixel) into either Class 1 (the rare target class, such as deforested land) or Class 2 (the background), the estimated true area of Class 1 is:

$$\hat{N}_{\bullet 1} = \left(1 - \frac{x_1}{n_1}\right) N_{1\bullet} + \frac{x_2}{n_2} N_{2\bullet}$$

where $x_1, x_2$ are misclassification counts from samples of size $n_1, n_2$ drawn from each stratum.

At each iteration the tool:
1. Tells you how many units to sample from each class
2. Asks you to label them and report misclassification counts
3. Updates the area estimate and its credible interval
4. Adaptively reallocates the next batch to minimise posterior variance
5. Stops once the credible interval falls within the target precision bounds

## Requirements

To use this tool, you must have Python available from the command line, with the `scipy` and `numpy` packages installed.

## Usage examples

### Work mode

```bash
python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.05 --batch 100
```

You will be prompted each iteration to label a batch of units and enter the misclassification counts.

Press `Ctrl+C` at any time to stop early. If the program is interrupted, the current progress is automatically saved to a checkpoint file so the run can be resumed later.

### Simulation mode

For testing, a random oracle can play the role of the human labeller:

```bash
python estimate_area.py --N1 5000 --N2 95000 --delta 0.1 --alpha 0.05 --batch 100 --simulate --true-p1 0.1 --true-p2 0.05
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--N1` | *(required)* | mapped unit count for Class 1 (rare class) |
| `--N2` | *(required)* | mapped unit count for Class 2 (background) |
| `--delta` | `0.1` | relative error target δ |
| `--alpha` | `0.05` | significance level α |
| `--batch` | `100` | number of units to label per iteration |
| `--simulate` | — | enable simulation mode |
| `--true-p1` | `0.1` | true Stratum 1 misclassification rate (simulation only) |
| `--true-p2` | `0.05` | true Stratum 2 misclassification rate (simulation only) |
| `--seed` | `666` | random seed (simulation only) |

Checkpoint files are automatically named using the run parameters, so restarting the program with the same arguments allows interrupted runs to be resumed.

## Example output

```
╔══════════════════════════════════════════════════════════╗
║   Rare-Class Area Estimation — Adaptive Stratified       ║
║   Sampling (Algorithm 1)                                 ║
╚══════════════════════════════════════════════════════════╝
  N1• = 5,000   N2• = 95,000   N = 100,000
  δ = 0.1   α = 0.05   batch size b = 100

...

  ┌─ Iteration 24 Results ───────────────────────────────────
  │  Cumulative samples : n1=274  n2=2126
  │  Misclassifications : x1=29  x2=112
  │  Estimate  N̂_{•1}  : 9,475.5
  │  95% credible interval : [8,615.4,  10,457.8]
  │  Precision target  : [8,614.1,  10,528.3]
  │  Target achieved?  : ✓ YES — stopping.
  └──────────────────────────────────────────────────

══════════════════════════════════════════════════════════
  FINAL ESTIMATE:  N̂_{•1} = 9,475.5
  95% credible interval: [8,615.4,  10,457.8]
  Total samples used: n1=274, n2=2126  (total=2400)
══════════════════════════════════════════════════════════
```

## What to enter when prompted

At each iteration you are asked to sample the specified number of units from each stratum and manually verify their true class labels:

- **x1** — how many of the Stratum 1 units are actually Class 2 (false positives)
- **x2** — how many of the Stratum 2 units are actually Class 1 (false negatives)

In a remote sensing workflow these would typically be verified against high-resolution imagery or field data.

## Precision criterion

Sampling continues until the (1−α) credible interval $[\hat{N}^L_{\bullet 1}, \hat{N}^U_{\bullet 1}]$ satisfies:

$$[\hat{N}_{\bullet 1}^L,\ \hat{N}_{\bullet 1}^U]\ \subseteq\ \left[\frac{\hat{N}_{\bullet 1}}{1+\delta},\ \frac{\hat{N}_{\bullet 1}}{1-\delta}\right]$$

i.e. the interval is entirely within δ relative precision of the point estimate.

## Checkpointing and resuming runs

The program automatically saves progress after each completed iteration to a checkpoint file.

The checkpoint filename encodes the run parameters. For example:

```
estimate_area_N1=5000_N2=95000_delta=0.1_alpha=0.05_b=100.checkpoint.json
```

This allows long labeling sessions to be interrupted and resumed later.

If you restart the program with the **same parameters**, it will detect the checkpoint file and ask whether to resume:

```
Found interrupted run checkpoint:
  estimate_area_N1=5000_N2=95000_delta=0.1_alpha=0.05_b=100.checkpoint.json  
Resume from saved state? [y/n]
```

- `y` resumes the run from the last completed iteration.
- `n` deletes the checkpoint and starts a new run.

When the algorithm reaches the final estimate and exits normally, the checkpoint file is automatically deleted.

