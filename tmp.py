import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(123)
odd_number = 5
seeded = torch.randn(odd_number)
print(seeded)