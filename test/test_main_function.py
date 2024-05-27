from slim.main_gp import gp

datasets = ["toxicity"]
n_runs = 1
pop_size = 100
n_iter = 100
p_xo = 0.8

gp(datasets=datasets, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter, p_xo=p_xo)