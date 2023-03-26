import torch

REALTYPE = torch.float32
COMPLEXTYPE = torch.complex64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")