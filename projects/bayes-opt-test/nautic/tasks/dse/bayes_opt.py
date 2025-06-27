import pandas as pd
from bayes_opt import BayesianOptimization
from nautic import taskx

class BayesOpt:
    @taskx
    def bayesian_opt(ctx):
        bo = ctx.bayes_opt
        bo.score = None
        log = ctx.log

        if bo.engine is None:
            bo.iteration = 0
            bo.summary = []

            pbounds = { }
            tune_vals = { }
            tune_space = { }
            tune_type = { }
            bo.control.params = { }
            for key in bo.tunable.model_fields:
                opt = getattr(bo.tunable, key)
                if isinstance(opt.space.get(), list):
                    pbounds[key] = (0, len(opt.space.get()) - 0.001)
                    tune_type[key] = 'categorical'
                elif isinstance(opt.space.get(), tuple) and len(opt.space.get()) == 2:
                    pbounds[key] = opt.space.get()
                    tune_type[key] = 'continuous'
                else:
                    raise ValueError(f"Unsupported space type for {key}: {type(opt.space.get())}")

                tune_vals[key] = opt.value
                tune_space[key] = opt.space

            bo.control.params['values'] = tune_vals
            bo.control.params['space'] = tune_space
            bo.control.params['type'] = tune_type

            bo.control.metrics = {}
            metrics_values = {}
            for key in bo.metrics.model_fields:
                metrics_values[key] = getattr(bo.metrics, key)
            bo.control.metrics['values'] = metrics_values

            score_weights = { }
            for key in bo.score_weights.model_fields:
                score_weights[key] = getattr(bo.score_weights, key)
            bo.control.metrics['score_weights'] = score_weights

            bo.engine = BayesianOptimization(
                f = None,
                pbounds=pbounds,
                random_state=bo.seed.get(),
                allow_duplicate_points=True,
                verbose=0
            )


            # Initial random points
            for _ in range(1):
                bo.control.suggests = dict(zip(pbounds.keys(),
                                               bo.engine._space.random_sample()))
        else:
            bo.iteration += 1
            engine = bo.engine
            score = 0
            for key in bo.control.metrics['values']:
                metric_value = bo.control.metrics['values'][key].get()
                base_value = bo.control.metrics['score_weights'][key].base
                weight_value = bo.control.metrics['score_weights'][key].weight

                score += float(metric_value / base_value) * float(weight_value)

            bo.score = score
            engine.register(params=bo.control.suggests,
                            target=bo.score)
            bo.control.suggests = bo.engine.suggest()

        summary = { 'iteration': bo.iteration,
                    'score': "n/a" if bo.score is None else round(bo.score, 4)}

        # set the parameters for other tasks
        for key, value in bo.control.suggests.items():
            idx = int(value)
            param_type = bo.control.params['type'][key]
            if param_type == 'categorical':
                metric_val = bo.control.params['space'][key].get()[idx]
            else: # continuous
                metric_val = value
            summary[key] = metric_val
            bo.control.params['values'][key].set(metric_val)

        bo.summary.append(summary)
        log.artifact(key='bayes-iteration-summary',
                     table=bo.summary)


        bo.terminate = not (bo.iteration < bo.num_iter)


