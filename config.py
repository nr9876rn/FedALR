import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='fashionmnist', help='dataset used for training')
    # parser.add_argument('--partition', type=str, default='noniid-skew', help='the data partitioning strategy')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=70, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=37, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="../Data/Dataset/", help="Data directory")
    parser.add_argument('--beta', type=float, default=1,
                        help='Thce parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default=2,
                        help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--lambda_1', type=float, default=0.01, help='fedprox hyperparameter')
    parser.add_argument('--fedamp_lambda', type=float, default=1.0, help='fedprox hyperparameter')
    parser.add_argument('--eps1', type=float, default=2.0, help='The hyper-parameter to control clustering (CFL)')
    parser.add_argument('--eps2', type=float, default=2.5, help='The hyper-parameter to control clustering (CFL)')
    parser.add_argument('--lambda_ewc', type=float, default=0.1, help='The hyper-parameter of EWC')
    parser.add_argument('--task_emb_dim', type=float, default=32, help='The hyper-parameter of FedWeIT')
    parser.add_argument('--lam', type=float, default=0.01, help="Hyper-parameter in the objective of pFedGraph")

    args = parser.parse_args()
    cfg = dict()
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fashionmnist', 'yahoo'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    elif args.dataset == 'sogounews':
        cfg['classes_size'] = 5
    elif args.dataset == 'dbpedia':
        cfg['classes_size'] = 14

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args, cfg
