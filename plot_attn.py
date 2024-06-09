import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str,
                    default="")
args = parser.parse_args()
filename = args.filename
score = torch.load(filename)
matrix = score[0, 0]
matrix = torch.where(
        torch.isinf(matrix),
        torch.full_like(matrix, 0),
        matrix)

sns.heatmap(matrix.detach().cpu().numpy().tolist(), center=0,)
plt.ylabel('query',fontsize=13)
plt.xlabel('key',fontsize=13)
plt.tight_layout()
plt.savefig("plot_fig/" + filename.split('.')[0] + '.pdf')
