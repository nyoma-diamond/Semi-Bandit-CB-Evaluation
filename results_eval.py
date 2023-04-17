import glob
import os
from enum import Enum

import numpy as np
import pandas as pd

from game_data import parse_identifier

class Measure(Enum):
    RAW = ''
    ERROR = ' Payoff/Regret Error'
    REG_CORRELATION = ' Regret Correlation'
    PAY_CORRELATION = ' Payoff Correlation'


def update_mats(mats, payoff, regret, target_alg, opp_alg):
    fmt = '{:.2f}'
    if measure not in [Measure.REG_CORRELATION, Measure.PAY_CORRELATION]:
        fmt += '\\pm' if get_latex else '±'
        fmt += '{:.2f}'
    if get_latex:
        fmt = f'${fmt}$'

    if measure == Measure.RAW:
        received_payoff = payoff['True Max'] - regret['True Max']
        mats['Received Payoff'].at[target_alg, opp_alg] = fmt.format(received_payoff.mean(), received_payoff.std())
        mats['Observable Expected'].at[target_alg, opp_alg] = fmt.format(regret['Observable Expected'].mean(), regret['Observable Expected'].std())
        mats['Observable Max'].at[target_alg, opp_alg] = fmt.format(regret['Observable Max'].mean(), regret['Observable Max'].std())
        mats['Supremum'].at[target_alg, opp_alg] = fmt.format(regret['Supremum'].mean(), regret['Supremum'].std())
    elif measure == Measure.ERROR:
        df = pd.DataFrame()
        df['Observable Expected'] = regret['Observable Expected'] - regret['True Expected']
        df['Observable Max'] = regret['Observable Max'] - regret['True Max']
        df['Supremum'] = regret['Supremum'] - regret['True Max']

        mean = lambda s: (s**2).mean() ** 0.5

        mats['Observable Expected'].at[target_alg, opp_alg] = fmt.format(mean(df['Observable Expected']), df['Observable Expected'].std())
        mats['Observable Max'].at[target_alg, opp_alg] = fmt.format(mean(df['Observable Max']), df['Observable Max'].std())
        mats['Supremum'].at[target_alg, opp_alg] = fmt.format(mean(df['Supremum']), df['Supremum'].std())
    elif measure == Measure.REG_CORRELATION:
        mats['Observable Expected'].at[target_alg, opp_alg] = fmt.format(regret['Observable Expected'].corr(regret['True Expected']))
        mats['Observable Max'].at[target_alg, opp_alg] = fmt.format(regret['Observable Max'].corr(regret['True Max']))
        mats['Supremum'].at[target_alg, opp_alg] = fmt.format(regret['Supremum'].corr(regret['True Max']))
    elif measure == Measure.PAY_CORRELATION:
        mats['Observable Expected'].at[target_alg, opp_alg] = fmt.format(payoff['Observable Expected'].corr(payoff['True Expected']))
        mats['Observable Max'].at[target_alg, opp_alg] = fmt.format(payoff['Observable Max'].corr(payoff['True Max']))
        mats['Supremum'].at[target_alg, opp_alg] = fmt.format(payoff['Supremum'].corr(payoff['True Max']))



def print_mats(mats, player, opp, T, K, A_resources, B_resources):
    tables = []

    for metric, mat in mats.items():
        if measure != Measure.RAW and metric == 'Received Payoff':
            continue

        if get_latex:
            caption = metric + measure.value
            table_str = mat.style.to_latex(column_format='rcccc',
                                           environment='subtable',
                                           position='h',
                                           position_float='centering',
                                           hrules=True,
                                           caption=caption)
            tables.append(table_str)
        else:
            print()
            print(player, K, A_resources, B_resources, metric)
            print(mat)

    if get_latex:
        big_table = '\\begin{table}[htb!p]\n' \
                    + '\n\\bigskip\n\n'.join(tables) \
                    + f'\\caption*{{Empirical results focusing on player {player} (rows) versus player {opp} (columns) for games with $T={T}$, $K={K}$, $N_A={A_resources}$, and $N_B={B_resources}$.}}\n' \
                    + '\\end{table}'

        big_table = big_table.replace('MARA', '\\ttsc{MARA}')\
            .replace('CUCB_DRA', '\\ttsc{CUCB-DRA}')\
            .replace('Edge', '\\ttsc{Edge}')\
            .replace('Random_Allocation', 'Random') \
            .replace('$nan$', '-') \
            .replace('[h]', '[h]{\\textwidth}')

        print('\n\n%==================================================\n%==================================================\n%==================================================\n\n')
        print(big_table)



in_dir = r'./results/**/*.npy'
column_order = ['True Expected', 'Observable Expected', 'True Max', 'Observable Max', 'Supremum']
measure = Measure.PAY_CORRELATION
get_latex = True

data = {}
# algorithms = set()
algorithms = ['MARA', 'CUCB_DRA', 'Edge', 'Random_Allocation'] # hard coding for convenient ordering

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

    # algorithms.add(game.A_algorithm)
    # algorithms.add(game.B_algorithm)

    results = np.load(path)
    data[game.T][game.K][game.A_resources][game.B_resources][game.A_algorithm][game.B_algorithm] = results

# algorithms = list(algorithms)


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
                print_mats(B, 'B', 'A', T, K, A_resources, B_resources)