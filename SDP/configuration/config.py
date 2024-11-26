import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument("--mode", type=str, default="er", help="Select CIL method")
    parser.add_argument("--dataset", type=str, default="cifar10", help="[cifar10, cifar100, tinyimagenet, imagenet]")
    parser.add_argument("--sigma", type=int, default=10, help="Sigma of gaussian*100")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat period")
    parser.add_argument("--init_cls", type=int, default=100, help="Percentage of classes already present in first period")
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument("--memory_size", type=int, default=500, help="Episodic memory size")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    # Model and Train
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--temp_batchsize", type=int, default=None, help="temporary batch size, for true online")
    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'randaug'],
                        help="Additional train transforms [cutmix, cutout, randaug]")

    # Data Loading
    parser.add_argument("--data_dir", type=str, help="location of the dataset")
    parser.add_argument("--n_worker", type=int, default=4, help="The number of workers")

    # Speeding up
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu.")
    parser.add_argument("--use_kornia", type=bool, default=True, help="use kornia")

    # Evaluation
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--f_period", type=int, default=10000, help="Period for measuring KLR and KGR")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    # Save
    parser.add_argument("--note", type=str, help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")

    # GDumb
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for GDumb")

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1,
                        help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # BiC
    parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")

    # AGEM
    parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    # SDP
    parser.add_argument('--sdp_mean', type=float, default=0.5, help='mean of dma weights, in period')
    parser.add_argument('--sdp_var', type=float, default=0.75, help='variance ratio (var/mean^2) of dma weights')

    args = parser.parse_args()
    return args
