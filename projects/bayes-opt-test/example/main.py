import os.path as osp
import sys
import random
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)
from nautic import Context

print(f"Root path: {root_path}")
ctx = Context.create('bayes.yml',
                     path=osp.join(root_path, 'nautic', 'tasks'))

engine = ctx.engine

# setup experiment (check bayes.yml for details)
ctx.seed = 42
ctx.params.p1_space = [0, 1, 2, 3, 4]
ctx.params.p2_space = [-3.5 + i * 0.1 for i in range(6)]

print(f"Parameter 1 space: {ctx.params.p1_space}")
print(f"Parameter 2 space: {ctx.params.p2_space}")

ctx.bayes_opt.num_iter = 4

while True:
    engine.dse.bayesian_opt()
    if ctx.bayes_opt.terminate:
        break

    prev_score = ctx.bayes_opt.score
    if prev_score is not None:
        prev_score = round(prev_score, 2)

    print(f"Iteration: {ctx.bayes_opt.iteration} | previous score: {prev_score}")
    print(f"+--- p1: {ctx.params.p1}, p2: {ctx.params.p2}")

    ### we did some model optimization
    #### we now evaluate the model, let's have some random values for w, x, y, z

    ctx.eval.w = round(random.uniform(0, 1), 2)
    ctx.eval.x = round(random.uniform(-1, 1), 2)
    ctx.eval.y = round(random.uniform(1, 2), 2)
    ctx.eval.z = round(random.uniform(-43, 532))

    print(f"+--- Evaluation results: w: {ctx.eval.w}, x: {ctx.eval.x}, "
          f"y: {ctx.eval.y}, z: {ctx.eval.z}")




