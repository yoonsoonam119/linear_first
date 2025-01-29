import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', **{'size':11})


def read_tps(args, dir, filename='report'):
    acc_trs = []
    acc_tes = []
    with open(f'{dir}/{filename}', 'r') as f:
        for line in f:
            acc_tr, acc_te, loss_tr, loss_te = [float(x) for x in line.strip().split()]
            acc_trs.append(acc_tr)
            acc_tes.append(acc_te)

    return acc_trs, acc_tes

def plot_accs(name, xlim=None):
    plt.figure(figsize=(4,3))
    data = np.load(f'./data/grokking_{name}.npy')
    acc_trs = data[0]
    acc_tes = data[1]
    if xlim is not None:
        acc_trs = acc_trs[:xlim]
        acc_tes = acc_tes[:xlim]
    plt.plot(np.arange(len(acc_trs))+1, acc_trs, label=r'$\mathrm{train}$', linestyle='solid', color=f'C0')
    plt.plot(np.arange(len(acc_tes))+1, acc_tes, label=r'$\mathrm{test}$', linestyle='solid', color=f'C1')
    plt.xscale('log')
    plt.xticks([1,10,100,1000])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(0,1.05)
    plt.xlabel(r'$\mathrm{Epochs}$')
    plt.ylabel(r'$\mathrm{Accuracy}$')
    plt.legend(loc='lower right', handlelength=1)
    plt.savefig(f'plot/11_{name}',dpi=300, bbox_inches='tight')
    plt.savefig(f'plot/11_{name}.pdf',format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--subfig", help="sub_fig", type=str, default="a")
    args = parser.parse_args()
    xlim = 100
    plot_accs(args.subfig, xlim=None)
