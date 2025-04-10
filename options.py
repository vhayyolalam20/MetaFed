import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MetaFed Configuration")
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR-10'])
    parser.add_argument('--iid', type=bool, default=False, help='Use IID (True) or Non-IID (False) distribution')
    parser.add_argument('--model', default='SimpleMLP')
    parser.add_argument('--selection_id', type=int, default=4, help='Selection strategy (1-6)')
    parser.add_argument('--aggregation', default='BWO', choices=['FedAvg', 'BWO'])
    parser.add_argument('--fitness', default='validation_accuracy')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', default='cpu')  # can be also "cuda"
    parser.add_argument('--sc', type=float, default=0.3)
    parser.add_argument('--lambda_perf', type=float, default=0.3)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--mutation_rate', type=float, default=0.3)
    parser.add_argument('--procreation_ratio', type=float, default=0.8)
    parser.add_argument('--cannibalism_ratio', type=float, default=0.5)
    return parser.parse_args()
