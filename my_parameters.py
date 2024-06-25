from my_args import my_args
import torch


default_dtype_torch = torch.float32
# device = torch.device("cuda:0"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# physical parameters:
beta_final = float(my_args.beta)
ham = "fm"

# network parameters:
net_depth = 2
net_width = 1

bias = True
z2 = bool(my_args.Z2)

epsilon = 1e-8

# learning parameters
max_step: int = my_args.max_step
beta_anneal: float = my_args.beta_anneal  # 0.996

batch_size: int = 2**my_args.batch_2_pow
print_step: int = 100
lr: float = my_args.lr

