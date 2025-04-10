import copy

import torch
from src.datasets import get_dataset
from src.client import Client
from src.utils import (
    get_client_selector,
    get_dataset_and_splits,
    get_model,
    get_fitness_function,
    get_aggregation_function,
)

from torch.utils.data import DataLoader, Subset

class MetaFed:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset_name = args.dataset
        self.selection_id = args.selection_id
        self.aggregation = args.aggregation
        self.fitness_name = args.fitness
        self.model_name = args.model
        self.sc = args.sc
        self.lambda_perf = args.lambda_perf

        self.clients = []
        self.performance_history = []

        # Load dataset, model, fitness function, and selector
       # self.train_data, self.val_data, self.test_data = get_dataset_and_splits(args.dataset)
        self.train_data, self.val_data, self.test_data, self.user_groups = get_dataset_and_splits(args)

        self.val_loader = DataLoader(self.val_data, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)

        self.model_fn = lambda: get_model(args.model, args.dataset, args.device)
        self.global_model = self.model_fn()
        self.selector = get_client_selector(self.selection_id)
        self.fitness_func = lambda weights: get_fitness_function(self.fitness_name)(
            weights, self.model_fn, self.val_loader, self.device
        )
        self.aggregate = get_aggregation_function(self.aggregation)

        self._create_clients()
   
        print("=" * 60)
        print(" MetaFed Training Setup")
        print(f"MetaFed initialized with:")
        print(f" Dataset           : {self.dataset_name}")
        print(f" Data Distribution : {'IID' if self.args.iid else 'Non-IID'}")
        print(f" Model             : {self.model_name}")
        print(f" Clients           : {self.args.num_clients}")
        print(f"  Rounds            : {self.args.rounds}")
        print(f"  Local Epochs      : {self.args.local_epochs}")
        print(f" Aggregation       : {self.aggregation}")
        print(f"  Fitness Function  : {self.fitness_name}")
        print(f"  Selection Strategy: {self.selection_id} - {self.selector.__name__}")
        print(f"  Clients: {len(self.clients)}, Rounds: {self.args.rounds}, Epochs: {self.args.local_epochs}")
        print("=" * 60)

    # def _create_clients(self):
    #     from torch.utils.data import random_split
    #     num_clients = self.args.num_clients
    #     partition_size = len(self.train_data) // num_clients
    #     for i in range(num_clients):
    #         indices = list(range(i * partition_size, (i + 1) * partition_size))
    #         subset = Subset(self.train_data, indices)
    #         self.clients.append(Client(i, subset, self.args))
    
    def _create_clients(self):
        # train, val, test, user_groups = get_dataset(
        #     self.args.dataset,
        #     iid=(self.args.selection_id in [1, 2, 3]),  # can be customized
        #     num_clients=self.args.num_clients
        # )
        # train, val, test, user_groups = get_dataset(
        #     name=self.args.dataset,
        #     iid=self.args.iid,
        #     num_clients=self.args.num_clients
        # )

        self.val_loader = DataLoader(self.val_data, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)

        for cid in range(self.args.num_clients):
            indices = list(self.user_groups[cid])
            client_data = Subset(self.train_data, indices)
            self.clients.append(Client(cid, client_data, self.args))


    def train(self):
        for rnd in range(self.args.rounds):
            print(f"\n--- Round {rnd + 1} ---")

            selector_kwargs = {
                "sc": self.sc,
                "round_num": rnd,
                "total_rounds": self.args.rounds,
                "perf_hist": self.performance_history,
                "lambda_perf": self.lambda_perf
            }

            selected_clients = self.selector(self.clients, **selector_kwargs)
            selected_ids = [c.cid for c in selected_clients]
            print(f"> Selected {len(selected_clients)} clients: {selected_ids}")

            local_weights = []
            for client in selected_clients:
                update = client.train(self.global_model)
                local_weights.append(update)

            if self.aggregation == 'BWO':
                best_weights, fitness_log = self.aggregate(
                    local_weights,
                    fitness_func=self.fitness_func,
                    generations=self.args.generations,
                    mutation_rate=self.args.mutation_rate,
                    procreation_ratio=self.args.procreation_ratio,
                    cannibalism_ratio=self.args.cannibalism_ratio,
                    model_fn=self.model_fn,
                    val_loader=self.val_loader,
                    device=self.device
                )
                self.global_model.load_state_dict(best_weights)
                print(f"> Max Fitness (val acc): {fitness_log[-1]:.4f}")
            else:
                new_weights = self.aggregate(local_weights)
                self.global_model.load_state_dict(new_weights)

            acc = self.evaluate()
            self.performance_history.append(acc)
            print(f"> Global Test Accuracy: {acc:.2f}%")

    def evaluate(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.global_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total
