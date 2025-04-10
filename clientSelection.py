# import random
# import numpy as np

# def select_clients_random_constant(clients, sc, **kwargs):
#     num_selected = max(1, int(len(clients) * sc))
#     return random.sample(clients, num_selected)

# def select_clients_random_dynamic_round(clients, sc, round_num, total_rounds, **kwargs):
#     frac = max(0.1, 1 - round_num / total_rounds)
#     num_selected = max(1, round(len(clients) * sc * frac))
#     return random.sample(clients, num_selected)

# def select_clients_random_dynamic_performance(clients, sc, perf_hist, lambda_perf, **kwargs):
#     if len(perf_hist) < 2:
#         return select_clients_random_constant(clients, sc)
#     delta_prev, delta_curr = perf_hist[-2], perf_hist[-1]
#     frac = max(0.1, 1 - lambda_perf * (delta_prev - delta_curr) / max(delta_prev, 1e-6))
#     num_selected = max(1, round(len(clients) * sc * frac))
#     return random.sample(clients, num_selected)

# def select_clients_rws_constant(clients, sc, **kwargs):
#     scores = np.array([c.evaluate_global() for c in clients])
#     probs = scores / np.sum(scores)
#     num_selected = max(1, round(len(clients) * sc))
#     return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))

# def select_clients_rws_dynamic_round(clients, sc, round_num, total_rounds, **kwargs):
#     frac = max(0.1, 1 - round_num / total_rounds)
#     scores = np.array([c.evaluate_global() for c in clients])
#     probs = scores / np.sum(scores)
#     num_selected = max(1, round(len(clients) * sc * frac))
#     return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))

# def select_clients_rws_dynamic_performance(clients, sc, perf_hist, lambda_perf, **kwargs):
#     if len(perf_hist) < 2:
#         return select_clients_rws_constant(clients, sc)
#     delta_prev, delta_curr = perf_hist[-2], perf_hist[-1]
#     frac = max(0.1, 1 - lambda_perf * (delta_prev - delta_curr) / max(delta_prev, 1e-6))
#     scores = np.array([c.evaluate_global() for c in clients])
#     probs = scores / np.sum(scores)
#     num_selected = max(1, round(len(clients) * sc * frac))
#     return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))

## The above and below are the same, the only difference is 
# num_selected = max(1, round(...)) >>>> num_selected = max(3, round(...)) 

import random
import numpy as np

def select_clients_random_constant(clients, sc, **kwargs):
    num_selected = max(3, int(len(clients) * sc))
    return random.sample(clients, num_selected)

def select_clients_random_dynamic_round(clients, sc, round_num, total_rounds, **kwargs):
    frac = max(0.1, 1 - round_num / total_rounds)
    num_selected = max(3, round(len(clients) * sc * frac))
    return random.sample(clients, num_selected)

def select_clients_random_dynamic_performance(clients, sc, perf_hist, lambda_perf, **kwargs):
    if len(perf_hist) < 2:
        return select_clients_random_constant(clients, sc)
    delta_prev, delta_curr = perf_hist[-2], perf_hist[-1]
    frac = max(0.1, 1 - lambda_perf * (delta_prev - delta_curr) / max(delta_prev, 1e-6))
    num_selected = max(3, round(len(clients) * sc * frac))
    return random.sample(clients, num_selected)

def select_clients_rws_constant(clients, sc, **kwargs):
    scores = np.array([c.evaluate_global() for c in clients])
    probs = scores / np.sum(scores)
    num_selected = max(3, round(len(clients) * sc))
    return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))

def select_clients_rws_dynamic_round(clients, sc, round_num, total_rounds, **kwargs):
    frac = max(0.1, 1 - round_num / total_rounds)
    scores = np.array([c.evaluate_global() for c in clients])
    probs = scores / np.sum(scores)
    num_selected = max(3, round(len(clients) * sc * frac))
    return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))

def select_clients_rws_dynamic_performance(clients, sc, perf_hist, lambda_perf, **kwargs):
    if len(perf_hist) < 2:
        return select_clients_rws_constant(clients, sc)
    delta_prev, delta_curr = perf_hist[-2], perf_hist[-1]
    frac = max(0.1, 1 - lambda_perf * (delta_prev - delta_curr) / max(delta_prev, 1e-6))
    scores = np.array([c.evaluate_global() for c in clients])
    probs = scores / np.sum(scores)
    num_selected = max(3, round(len(clients) * sc * frac))
    return list(np.random.choice(clients, size=num_selected, replace=False, p=probs))
