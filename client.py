import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy

class Client:
    def __init__(self, cid, data, args):
        self.cid = cid
        self.data = data
        self.args = args
        self.history = []
        self.data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    def train(self, global_model):
        model = copy.deepcopy(global_model).to(self.args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        model.train()

        for _ in range(self.args.local_epochs):
            for x, y in self.data_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def evaluate_global(self):
        # Dummy score, override if needed
        return 1.0
