import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
group = parser.add_argument_group("physics parameters")

group.add_argument(
    "-b", "--beta", type=float, default=0.4406868, help="beta = 1 / k_B T"
)

group.add_argument("-L", "--L", type=int, default=8, help="horizontal size")
group.add_argument(
    "-k",
    "--k",
    type=int,
    default=1,
    help="Y/L, multiplicative constant defining vertical size",
)
group.add_argument(
    "-s",
    "--shift",
    type=int,
    default=0,
    help="Defines shift of the boundary between two subsystems of two replicas. ",
)
group.add_argument("-Z2", type=int, default=1, help="Z2 symmetry on")

group2 = parser.add_argument_group("model parameters + program options")
group2.add_argument(
    "-sp",
    "--show_plots",
    action="store_false",
    help="set if plots are to be shown",
)
group2.add_argument(
    "-pp", "--path_plots", type=str, default="./plots", help="Place for saved plots"
)
group2.add_argument(
    "-ms",
    "--max_step",
    type=int,
    default=1000,
    help="Number of training step to be performed during training",
)
group2.add_argument("-lr", type=float, default=0.001, help="Learning rate")
group2.add_argument(
    "-ba", "--beta_anneal", type=float, default=0, help="Beta annealing factor"
)
group2.add_argument(
    "-b2^",
    "--batch_2_pow",
    type=int,
    default=10,
    help="index of the power of 2 defining batch size. Ex. if desired batch size is 32 = 2**5, set batch_2_pow=5",
)
group2.add_argument(
    "-pb",
    "--pretrained_beta_shift",
    type=float,
    default=0,
    help="Load pretrained network for beta shifted by given value",
)
group2.add_argument(
    "-ps",
    "--path_save",
    type=str,
    default="./saved_states",
    help="Place for saved models",
)
group2.add_argument(
    "-pd", "--path_data", type=str, default="./data", help="Place for generated data"
)
group2.add_argument(
    "-ol",
    "--overload",
    action="store_true",
    help="If set, the existing model will be overwrite (instead of continued learning)",
)

group2.add_argument(
    "-i",
    "--import_pretrained",
    action="store_true",
    help="""For one_replica: the script will try to import pretraied model from the network with k-1, or L/2 if k==0.
    For two_replicas: if pretrained_beta_shift is set, import of the model with the same size but different beta, 
    else if shift=0, it tries to import pretrained network from one_replica, else if shift != 0, it tries to import pretrained netword from shift=0.
    Failure will stop the training"""
)

my_args = parser.parse_args()
