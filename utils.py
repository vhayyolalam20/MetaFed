from src.datasets import get_dataset
from src.models import get_model
from src.aggregation import fed_avg, black_widow_optimization
from src.fitness import validation_accuracy, weight_norm
from src.clientSelection import *


def get_client_selector(selection_id):
    return [
        select_clients_random_constant,
        select_clients_random_dynamic_round,
        select_clients_random_dynamic_performance,
        select_clients_rws_constant,
        select_clients_rws_dynamic_round,
        select_clients_rws_dynamic_performance
    ][selection_id - 1]

def get_fitness_function(name):
    return validation_accuracy if name == 'validation_accuracy' else weight_norm

def get_aggregation_function(name):
    return fed_avg if name == 'FedAvg' else black_widow_optimization

# def get_dataset_and_splits(name):
#     return get_dataset(name)

def get_dataset_and_splits(args):
    return get_dataset(args
        # name=args.dataset,
        # iid=args.iid,
        # num_clients=args.num_clients
    )
