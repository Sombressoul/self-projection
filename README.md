# SelfProjection Module for PyTorch
- [SelfProjection Module for PyTorch](#selfprojection-module-for-pytorch)
  - [Overview](#overview)
  - [Approach](#approach)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Evaluation](#evaluation)
  - [Future Directions](#future-directions)
  - [Contribution](#contribution)

## Overview
The `SelfProjection` module is a PyTorch-based neural network layer designed to transform and project high-dimensional data. It is particularly useful in contexts requiring sophisticated analysis and representation of feature relationships, such as outputs from Transformer models.

## Approach
The `SelfProjection` module employs a dual projection mechanism to process input tensors, capturing different perspectives of the data. Key aspects include:

- **Original and Permuted Projections**: The module processes the input tensor in its original form and a permuted form, creating two distinct representations.
- **Relational Interference**: By computing relational matrices, the module captures the interplay between different projections, emphasizing the relationships between data dimensions.
- **Normalization**: Custom normalization steps, which involve mean subtraction and variance scaling similar to Layer Normalization, are applied to the projections, ensuring stable feature scaling and potentially improved model performance.
- **Trainable Parameters**: The module includes several trainable parameters, allowing it to learn optimal feature transformations during training.

## Installation

__Using pip:__

To install the `SelfProjection` module using pip, simply run the following command:

```bash
pip install self-projection
```

__From source:__

To install the `SelfProjection` module, clone this repository and import the module into your PyTorch project.

```bash
git clone https://github.com/Sombressoul/self-projection ./self_projection
python -m pip install -e ./self_projection
```

## Usage

Here's a simple example of how to use the `SelfProjection` with PyTorch:

```python
import torch
from self_projection import SelfProjection

# Define the input tensor dimensions and projection size
input_tensor = torch.randn((batch_size, sequence_length, embedding_dim))
size_projection = 128

# Initialize the SelfProjection module
self_projection = SelfProjection(size_input=input_tensor.size()[1::], size_projection=size_projection)

# Apply the module to the input tensor
projected, relations = self_projection(input_tensor)

print(projected.shape)
# >>> torch.Size([<batch_size>, 128, 128])
print(relations.shape)
# >>> torch.Size([<batch_size>, 128, 128])
```

## Evaluation

(deleted as outdated since 20.01.2024)

Pending.

## Future Directions

The ongoing development of the `SelfProjection` module is a component of a personal endeavor in the field of machine learning. Future plans include conducting experiments with more complex datasets to further assess and refine the module's capabilities. These steps are aimed at exploring a broader range of applications and enhancing the module's performance in diverse settings.

This module is a reflection of my interest in contributing to the machine learning community through individual efforts and is one aspect of a larger personal project dedicated to exploring innovative approaches in the field.

## Contribution

Contributions to the `SelfProjection` module are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.
