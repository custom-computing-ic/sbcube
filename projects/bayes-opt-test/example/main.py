import os.path as osp
import sys
import random
import numpy as np
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)
from nautic import Context


print(f"Root path: {root_path}")
ctx = Context.create('bayes.yml',
                     path=osp.join(root_path, 'nautic', 'tasks'))

engine = ctx.engine

# setup experiment (check bayes.yml for details)

ctx.params.pruned_space = (-10, 10)
ctx.params.locked_space = (-10, 10)
ctx.params.nfrozen_space = (-10, 10)

ctx.seed = 42
ctx.bayes_opt.num_iter = 10

while True:
    engine.dse.bayesian_opt()
    if ctx.bayes_opt.terminate:
        break

    prev_score = ctx.bayes_opt.score
    if prev_score is not None:
        prev_score = round(prev_score, 2)

    # unconstrained parameters
    pruned = ctx.params.pruned
    locked = ctx.params.locked
    nfrozen = ctx.params.nfrozen

    exp_vals = np.exp([pruned, locked, nfrozen])
    softmax_vals = exp_vals / np.sum(exp_vals)

    # constrainted parameters, the sum must be 1
    _pruned = round(softmax_vals[0], 3)
    _locked = round(softmax_vals[1], 3)
    _nfrozen = round(softmax_vals[2], 3)

    print(f"Iteration: {ctx.bayes_opt.iteration} | previous score: {prev_score}")
    print(f"+--- p: {_pruned}, "
          f" l: {_locked}, "
          f" nf: {_nfrozen}")

    ### we did some model optimization
    #### we now evaluate the model, let's have some random values for w, x, y, z

    ctx.eval.w = round(random.uniform(0, 1), 2)
    ctx.eval.x = round(random.uniform(-1, 1), 2)
    ctx.eval.y = round(random.uniform(1, 2), 2)
    ctx.eval.z = round(random.uniform(-43, 532))

    print(f"+--- Evaluation results: w: {ctx.eval.w}, x: {ctx.eval.x}, "
          f"y: {ctx.eval.y}, z: {ctx.eval.z}")




