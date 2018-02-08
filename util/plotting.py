import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def plot_data(data, value="AverageReturn", savename=None):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    g = sns.tsplot(data=data, time="Episode", value=value, unit="Unit", condition="Condition")
    g.set(xlim=(0, 3000))
    g.set(ylim=(0, 195))
    plt.legend(loc='best').draggable()
    fig = plt.gcf()
    fig.savefig('figures/{}.pdf'.format(savename), bbox_inches='tight')
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root, 'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']

            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
            )

            datasets.append(experiment_data)
            unit += 1

    return datasets


def plot_result(logdir_root, value, legend=None, savename=None):
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('logdir', nargs='*')
    # parser.add_argument('--legend', nargs='*')
    # parser.add_argument('--value', default='AverageReturn', nargs='*')
    # parser.add_argument('--value', default='AvgScoresFor100Episodes', nargs='*')
    # args = parser.parse_args()

    use_legend = False
    if legend is not None:
        assert len(legend) == len(logdir_root), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(logdir_root, legend):
            data += get_datasets(logdir, legend_title)
    else:
        # for logdir in logdir_root:
        # print(logdir_root)
        # print(logdir)
        data += get_datasets(logdir_root)

    if isinstance(value, list):
        values = value
    else:
        values = [value]
    for v in values:
        plot_data(data, value=v, savename=savename)
