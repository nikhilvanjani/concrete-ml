#!/usr/bin/env python
# coding: utf-8

# # Deep MNIST classifiers
# 
# ## Reproducing the results from **Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks**
# 
# This notebook replicates experiments from the paper [_Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks_](https://whitepaper.zama.ai/), published in 2021. This previous work did not use the Concrete compiler.
# This notebook reproduces the deep neural network architectures NN-20 and NN-50 from the whitepaper and shows how they are [quantized](https://docs.zama.ai/concrete-ml/explanations/quantization) using Post Training Quantization (PTQ) and compiled with Concrete to work on encrypted data.
# 
# We compare the original paper's findings with the results from the latest version of [Concrete ML](https://pypi.org/project/concrete-ml/). 
# 
# The results shown at the end of this notebook show a large speed-up with respect to the results in the whitepaper.

# In[1]:


import os
import random
import time
import warnings

import numpy as np
import torch
from torchvision import datasets, transforms
from utils_experiments import MEAN, STD, format_results_df, plot_dataset, torch_inference

from concrete.ml.torch.compile import compile_torch_model

warnings.filterwarnings("ignore", category=UserWarning)


# ## Architecture
# 
# All networks begin with a convolutional layer configured with `in_channel=1, out_channels=1, kernel_size=3, stride=1, padding_mode='replicate'`. 
# 
# This is followed by 20 linear layers of 92 neurones with ReLU activation for NN-20, and 50 layers for NN-50.

# In[2]:


DEVICE = "cpu"

# Input size, 28x28 pixels, a standard size for MNIST images
INPUT_IMG_SIZE = 28

# Batch size
BATCH_SIZE = 64

# Seed to ensure reproducibility
SEED = 42

# Whether the experiments are run on PC or other machines, like HP7C on AWS
MACHINE = f"{os.cpu_count()}-core machine"

# The timing and the accuracy recorded in the article
if os.cpu_count() > 48:
    PAPER_NOTES = {20: [21.17, 0.971], 50: [43.91, 0.947]}
else:
    PAPER_NOTES = {20: [115.52, 0.971], 50: [233.55, 0.947]}


# ## FP32 MNIST Neural Nerwork 

# In[3]:


FEATURES_MAPS = [
    # Convolution layer, with:
    # in_channel=1, out_channels=1, kernel_size=3, stride=1, padding_mode='replicate'
    ("C", 1, 1, 3, 1, "replicate"),
    ("R",),
    ("B", 1, 30),  # 2d batch-norm for 1 channel
]


# The article presents 3 neural network depths. In this notebook, we focus NN-20 and NN-50
# architectures. The parameter `nb_layers`: controls the depth of the NN.
def LINEAR_LAYERS(nb_layers: int, output_size: int):
    return (  # noqa: W503
        [
            ("L", INPUT_IMG_SIZE * INPUT_IMG_SIZE, 92),
            ("R",),
            ("B", 92),  # 1d batch norm
        ]  # noqa: W503
        + [  # noqa: W503
            ("L", 92, 92),
            ("R",),
            ("B", 92),  # 1d batch norm
        ]
        * (nb_layers - 3)  # noqa: W503
        + [("L", 92, output_size)]  # noqa: W503
    )


class Fp32MNIST(torch.nn.Module):
    """MNIST Torch model."""

    def __init__(self, nb_layers: int, output_size: int = 10):
        """MNIST Torch model.

        Args:
            nb_layers (int): Number of layers.
            output_size (int): Number of classes.
        """
        super().__init__()

        self.nb_layers = nb_layers
        self.output_size = output_size

        def make_layers(t):
            if t[0] == "C":
                # Workaround: stride=1, padding_mode='replicate' is replaced by
                # transforms.Pad(1, padding_mode="edge")
                return torch.nn.Conv2d(
                    in_channels=t[1],
                    out_channels=t[2],
                    kernel_size=t[3],
                )
            if t[0] == "L":
                return torch.nn.Linear(in_features=t[1], out_features=t[2])
            if t[0] == "R":
                return torch.nn.ReLU()
            if t[0] == "B":
                if len(t) == 2:
                    return torch.nn.BatchNorm1d(t[1])
                if len(t) == 3:
                    return torch.nn.BatchNorm2d(t[1])

            raise NameError(f"'{t}' not defined")

        self.features_maps = torch.nn.Sequential(*[make_layers(t) for t in FEATURES_MAPS])
        self.linears = torch.nn.Sequential(
            *[make_layers(t) for t in LINEAR_LAYERS(self.nb_layers, self.output_size)]
        )

    def forward(self, x):
        x = self.features_maps(x)
        x = torch.nn.Flatten()(x)
        x = self.linears(x)
        return x


# ## Load and pre-process the MNIST data-set
# 

# In[4]:


g = torch.Generator()
g.manual_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

train_transform = transforms.Compose(
    [  # Workaround: stride=1, padding_mode='replicate' is replaced by
        transforms.Pad(1, padding_mode="edge"),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ]
)
test_transform = transforms.Compose(
    [  # Workaround: stride=1, padding_mode='replicate' is replaced by
        transforms.Pad(1, padding_mode="edge"),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ]
)

train_dataset = datasets.MNIST(download=True, root="./data", train=True, transform=train_transform)

test_dataset = datasets.MNIST(download=True, root="./data", train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=g,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=g,
)

plot_dataset(train_loader)


# ## Model architecture
# 
# The following cell shows the architecture of the NN-20 model. It contains one convolutional layer and 19 linear ones. 

# In[5]:


fp32_mnist = Fp32MNIST(nb_layers=20).to(DEVICE)
print(fp32_mnist)


# ## Benchmark the models
# 
# In the next sections of this notebook, we apply the same experimental protocol to both NN-20 and NN-50 models.
# 
# To make a custom neural network FHE-compatible, it's necessary to quantize both the network and its inputs. Post-Training Quantization (PTQ) involves quantizes a pre-trained floating-point model directly, and does not require re-training. In Concrete ML, the compilation of the custom NNs is performed through `compile_torch_model` function
# 

# In the compilation step, the compiler requires a representative set of data,  named `data_calibration` below, to evaluate the maximum integer bit-width within the graph.

# In[6]:


data_calibration = next(iter(train_loader))[0]

results_cml = {}

for nb_layers in [20, 50]:

    fp32_mnist = Fp32MNIST(nb_layers=nb_layers).to(DEVICE)

    checkpoint = torch.load(
        f"./checkpoints/MNIST/MLP_{nb_layers}/fp32/MNIST_fp32_state_dict.pt", map_location=DEVICE
    )
    fp32_mnist.load_state_dict(checkpoint)
    fp32_mnist.eval()

    acc_test = torch_inference(fp32_mnist, test_loader, device=DEVICE)

    # The model is compiled through 'compile_torch_model' method
    # We use approximate rounding and a p_error value of 0.1
    # These values work well for neural networks which are robust to noise
    # in the computation of intermediate values.
    q_module = compile_torch_model(
        fp32_mnist.to(DEVICE),
        torch_inputset=data_calibration,
        n_bits=6,
        rounding_threshold_bits={"n_bits": 6, "method": "APPROXIMATE"},
        p_error=0.1,
    )

    fhe_timing = []
    y_predictions = []
    fhe_samples = 3

    # The model is evaluated through all the test data-set in 'simulation' mode
    for i, (data, labels) in enumerate(test_loader):

        data, labels = data.detach().cpu().numpy(), labels.detach().cpu().numpy()
        simulate_predictions = q_module.forward(data, fhe="simulate")
        y_predictions.extend(simulate_predictions.argmax(1) == labels)

        # Measure FHE latency on three samples and take the minimum
        if i <= fhe_samples:
            start_time = time.time()
            q_module.forward(data[0, None], fhe="execute")
            fhe_timing.append((time.time() - start_time))

    results_cml[nb_layers] = [acc_test, np.mean(y_predictions), np.min(fhe_timing)]

    print(
        f"Running NN-{nb_layers} on a {MACHINE} machine:"
        f"Accuracy in fp32 : {results_cml[nb_layers][0]:.3%} for the test set\n"
        f"Accuracy with FHE-simulation mode : {results_cml[nb_layers][1]:.3%} for the test set\n"
        f"FHE Latency on encrypted data : {results_cml[nb_layers][2]:.3f}s per encrypted sample.\n"
        f"Number of PBS: {q_module.fhe_circuit.statistics['programmable_bootstrap_count']}"
    )


# ## Conclusion
# 
# Here is a recap of the results obtained by running this notebook on a `m6i.metal` instance, compared to the results in the whitepaper [1].

# In[7]:


df, fmt = format_results_df(PAPER_NOTES, results_cml, "Our m6i.metal")
df.style.format(fmt)


# A pre-computed comparison to a `hpc7a.96xlarge` instance with 192 vCPU is also shown when running this notebook on a large server machine.

# In[8]:


# This benchmark was done on a hpc7 machine with 192-cores
BENCH_HPC7A = {20: [0.987, 0.959, 0.995], 50: [0.9745, 0.9477, 3.03]}

df, fmt = format_results_df(PAPER_NOTES, BENCH_HPC7A, "Our hpc7a.96xlarge")
df.style.format(fmt)


# As shown in the table above, on the `hpc7` instance, Concrete ML achieves a ~20x speed-up compared to the whitepaper.
