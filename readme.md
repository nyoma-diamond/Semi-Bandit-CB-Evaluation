# Evaluating Repeated Clonel Blotto Games with Semi-Bandit Feedback

### Author: N'yoma Diamond ([ORCiD](https://orcid.org/0000-0002-6468-1779), [LinkedIn](https://www.linkedin.com/in/nyoma-diamond/), [GitHub](https://github.com/nyoma-diamond))

#### Worcester Polytechnic Institute, Department of Data Science

Code for experiments and results analysis conducted in the Master's thesis of the same name.


## Requirements

### Python Version: 3.9.13

### Dependencies

- [NumPy](https://numpy.org/) (ver. 1.24.2)
- [pandas](https://pandas.pydata.org/) (ver. 1.5.1)
- [tqdm](https://github.com/tqdm/tqdm) (ver. 4.65.0)
- [dill](https://github.com/uqfoundation/dill) (ver. 0.3.6)
- [SciPy](https://scipy.org/) (ver. 1.9.3)


## File Organization

The files in this project are organized as follows:

- `pdgraph.py` contains code for decision graphs, decision generation, and payoff/regret metric computation.
- `run_games.py` contains code for simulating repeated colonel blotto games.
- `experiment.py` contains code for computing payoff/regret metrics on simulated games.
- `results_eval.py` contains code for cleaning simulated game data and computing error metrics.
- `game_data.py` contains a basic data structure for storing simulated game data.
- `algorithms/*` contains code for the algorithms used in our simulations.


## Simulating Games

Games are simulated using the code in `run_games.py`. Simulated games are serialized and saved using `dill`.


### Editable Parameters

`run_games.py` has a number of parameters that can be changed to generate different simulations:

 - `out_dir`: Output directory to save result to. `simulations/<time>` By default, where `<time>` becomes the time that the code was executed

 - `battlefields`: Array of battlefield values to simulate.

 - `resources`: Array of possible resources to give to each player

    **NOTE:** Resource matchups (i.e., how many resources does the draw-winning player have versus the draw-losing player) are generated such each matchup is a unique _combination_. I.e., 15-20 and 20-15 are not unique matchups (despite being different). This was done so the draw-losing player always has the same or more resources than the draw-winning player. If all _permutations_ are desired, change `combinations_with_replacement` to `product`.

 - `T`: The number of rounds to simulate up to.

 - `algorithms`: The algorithms to simulate and their parameters


## Computing Metrics

Games are simulated using the code in `experiment.py`. Per-round results are generated for each game and stored as NumPy `.npy` files.


### Editable Parameters

`experiment.py` has a number of parameters that can be changed to generate different simulations:

- `in_dir`: The input diretory to load simulated game data from. `./simulations/**/*` by default (this loads every non-directory file in `./simulations` and its subdirectories).

- `out_dir`: Output directory to save result to. `results/<time>` By default, where `<time>` becomes the time that the code was executed

- `chunksize`: The chunksize parameter for multiprocessing. We recommend at least 32 in order to improve efficiency and reduce the number of processes created.

- `max_parallel_games`: The number of games to compute metrics for in parallel. This is useful to minimize CPU idling when a game is almost, but not completely done being processed. If this value is 1 then computation may spend a lot of time waiting for other processes to finish. It is a good idea to have this be at least 2 so that another game is always ready to be processed

  **WARNING:** Setting this value very high can cause system problems/crashes due to extremely large resource usage. It is recommended that this value not exceed 4 as very little benefit is gained beyond that value

- `latex_output`: Change table output to print $\LaTeX$ formatted tables instead of DataFrames. `False` by default.

- `combine_anova_latex`: Combine full and reduced ANOVA model printouts into one table.

  **NOTE:** **_Requires_** `latex_output` to be `True`, does nothing otherwise. Overrides `only_reduced` if `True`


## Evaluating Results

Our metrics are evaluated using the code in `results_eval.py`. The metrics for each game are evaluated and summarized into either command-line printouts or $\LaTeX$ table code.


### Editable Parameters

`experiment.py` has a number of parameters that can be changed to generate different simulations:

- `in_dir`: The input diretory to load simulated game data from. `./results/**/*.npy` by default (this loads every non-directory file with the extension `.npy` in `./results` and its subdirectories).

- `measure`: The measure to compute and display results for:
  - `Measure.RAW`: The mean and standard deviation of the raw metrics: Received payoff, Observable Expected Regret, Observable Max Regret, and Supremum Regret.
  - `Measure.ERROR`: The RMSE and SDE of the computed estimation metrics.
  - `Measure.REG_CORRELATION`: The Pearson correlation between the estimated regret metrics and their true values.
  - `Measure.REG_CORRELATION`: The Pearson correlation between the estimated payoff metrics and their true values.
 
- `get_latex`: Change table output to print $\LaTeX$ formatted tables instead of DataFrames. `False` by default.
