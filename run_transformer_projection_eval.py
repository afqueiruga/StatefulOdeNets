import itertools
import glob
import os

import matplotlib.pylab as plt
import numpy as np

from flax import traverse_util
from flax.core import unfreeze

from continuous_net_jax.tools import count_parameters
from continuous_net_jax.convergence import *
from continuous_net_jax.basis_functions import *
from continuous_net_jax.basis_functions_plotting import *
from continuous_transformer.continuous_transformers import *
from continuous_transformer.sentence_tagger_tester import *
from continuous_transformer import input_pipeline

DIR = "/home/afq/tree_bank/german_hdt/"
paths = glob.glob(
    DIR + '/ContinuousTransformer_config=TransformerConfig_128,1,1,128,128*')

# Main text example.
# tt = TransformerTester(
#     datadir="../ud-treebanks-v2.8/UD_English-GUM/",
#     prefix='en_gum',
#     max_len=102)
# Supplementary material example.
tt = TransformerTester(datadir="../ud-treebanks-v2.8/UD_German-HDT/",
                       prefix='de_hdt',
                       max_len=91)

for path in paths:
    ct = ConvergenceTester(path, scope=globals())
    print(ct.eval_model)
    # Make the data for the compression plot, Figure 5a.
    tt.perform_project_and_infer(ct,
                                 ['piecewise_constant', 'fem_linear'],
                                 [1, 2, 3, 4, 8, 16, 24, 32, 48, 64],
                                 ['Euler'],
                                 [64])
    tt.perform_project_and_infer(ct,
                                 ['piecewise_linear'],
                                 [2, 4, 8, 16, 24, 32, 48, 64],
                                 ['Euler'],
                                 [64])

    # Make the data for the NT ablation plot, Figure 5b.
    tt.perform_project_and_infer(ct,
                                 ['piecewise_constant'],
                                 [1, 2, 3, 4, 8, 16, 24, 32, 48, 64],
                                 ['Euler'],
                                 [8, 16, 32, 64])
