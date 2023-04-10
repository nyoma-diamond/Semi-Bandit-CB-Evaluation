import glob
import os

import numpy as np
import pandas as pd

from game_data import parse_identifier

def update_mats(mats, payoff, regret, target_alg, opp_alg):
    fmt = '${:.3f}\\pm{:.3f}$' if get_latex else '{:.3f}±{:.3f}'
    received_payoff = payoff['True Max'] - regret['True Max']
    mats['Received Payoff'].at[target_alg, opp_alg] = fmt.format(received_payoff.mean(), received_payoff.std())

    obs_expected_diff = payoff['Observable Expected'] - payoff['True Expected']
    mats['Observable Expected'].at[target_alg, opp_alg] = fmt.format(obs_expected_diff.mean(), obs_expected_diff.std())

    obs_max_diff = payoff['Observable Max'] - payoff['True Max']
    mats['Observable Max'].at[target_alg, opp_alg] = fmt.format(obs_max_diff.mean(), obs_max_diff.std())

    supremum_diff = payoff['Supremum'] - payoff['True Max']
    mats['Supremum'].at[target_alg, opp_alg] = fmt.format(supremum_diff.mean(), supremum_diff.std())


def get_style(df):
    return df.style.set_table_styles([
        {'selector': 'toprule', 'props': ':hline;'},
        {'selector': 'midrule', 'props': ':hline;'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ])


def print_mats(mats, player, opp, T, K, target_resources, opp_resources):
    tables = []

    for metric, mat in mats.items():
        if get_latex:
            table_str = '\n\\begin{subtable}[h]{\\textwidth}\n' \
                        + '\\centering\n' \
                        + get_style(mat).to_latex(column_format='||c|cccc||') \
                        + f'\\caption{{{metric}}}\n' \
                        + '\\end{subtable}'
            tables.append(table_str)
        else:
            print()
            print(player, K, target_resources, opp_resources, metric)
            print(mat)

    if get_latex:
        big_table = '\\begin{table}[htb!p]' \
                    + '\n\n\\bigskip\n'.join(tables) \
                    + f'\n\\caption{{Empirical results focusing on player {player} (rows) vs player {opp} (columns) for games with $T={T}$ rounds, $K={K} battlefields$, $N_A={target_resources}$ player A resources, and $N_B={opp_resources}$ player B resources. (a) The mean and standard deviation of received payoff by player {player} over the course of the game. (b) The mean and standard deviation of the difference between True Expected Payoff/Regret and Observable Expected Payoff/Regret over the course of the game. (c) The mean and standard deviation of the difference between True Max Payoff/Regret and Observable Max Payoff/Regret over the course of the game. (d) The mean and standard deviation of the difference between True Max Payoff/Regret and Supremum Payoff/Regret over the course of the game.}}\n' \
                    + '\\end{table}'

        print('\n\n%==================================================\n%==================================================\n%==================================================\n\n')
        print(big_table)



in_dir = r'./results/**/*.npy'
column_order = ['True Expected', 'Observable Expected', 'True Max', 'Observable Max', 'Supremum']
get_latex = False

data = {}
algorithms = set()

for path in glob.glob(in_dir):
    game = parse_identifier(os.path.basename(path)[:-len('.npy')])

    if game.T not in data.keys():
        data[game.T] = {}
    if game.K not in data[game.T].keys():
        data[game.T][game.K] = {}
    if game.A_resources not in data[game.T][game.K].keys():
        data[game.T][game.K][game.A_resources] = {}
    if game.B_resources not in data[game.T][game.K][game.A_resources].keys():
        data[game.T][game.K][game.A_resources][game.B_resources] = {}
    if game.A_algorithm not in data[game.T][game.K][game.A_resources][game.B_resources].keys():
        data[game.T][game.K][game.A_resources][game.B_resources][game.A_algorithm] = {}

    algorithms.add(game.A_algorithm)
    algorithms.add(game.B_algorithm)

    results = np.load(path)
    data[game.T][game.K][game.A_resources][game.B_resources][game.A_algorithm][game.B_algorithm] = results

algorithms = list(algorithms)


for T in data.keys():
    for K in data[T].keys():
        for A_resources in data[T][K].keys():
            for B_resources in data[T][K][A_resources].keys():
                A = {
                    'Received Payoff': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Observable Expected': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Observable Max': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Supremum': pd.DataFrame(index=algorithms, columns=algorithms)
                }

                B = {
                    'Received Payoff': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Observable Expected': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Observable Max': pd.DataFrame(index=algorithms, columns=algorithms),
                    'Supremum': pd.DataFrame(index=algorithms, columns=algorithms)
                }

                for A_algorithm in data[T][K][A_resources][B_resources].keys():
                    for B_algorithm, results in data[T][K][A_resources][B_resources][A_algorithm].items():
                        A_payoff = pd.DataFrame(results[0,0], columns=column_order)
                        A_regret = pd.DataFrame(results[0,1], columns=column_order)

                        B_payoff = pd.DataFrame(results[1,0], columns=column_order)
                        B_regret = pd.DataFrame(results[1,1], columns=column_order)

                        update_mats(A, A_payoff, A_regret, A_algorithm, B_algorithm)
                        update_mats(B, B_payoff, B_regret, B_algorithm, A_algorithm)


                print_mats(A, 'A', 'B', T, K, A_resources, B_resources)
                print_mats(B, 'B', 'A', T, K, B_resources, A_resources)