# SelfProjection Module for PyTorch
- [SelfProjection Module for PyTorch](#selfprojection-module-for-pytorch)
  - [Overview](#overview)
  - [Approach](#approach)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Evaluation](#evaluation)
    - [Experimental Setup](#experimental-setup)
    - [Insights](#insights)
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

To install the `SelfProjection` module, clone this repository and import the module into your PyTorch project.

```bash
git clone https://github.com/Sombressoul/self-projection ./self_projection
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

The `SelfProjection` module has been evaluated using the MNIST dataset under various conditions to test its efficiency in spatial feature extraction and overall performance. 

[eval_mnist.py](eval_mnist.py) - contains an evaluation code for MNIST dataset. By default it is set to extreme conditions with heavy projection reduction (4x4) and high dropout rate (0.75).

### Experimental Setup
Two key experimental setups were employed:

1. **Heavy Reduction with High Dropout**:
   - Initial tests with a projection size of 4x4 and a high dropout rate of 0.75 demonstrated the module's capability to still achieve an accuracy of 53% on the MNIST test set. This setup was particularly challenging due to the substantial dimensionality reduction and high dropout rate, putting a stress test on the feature extraction abilities of `SelfProjection`.

2. **Standard Conditions**:
   - Further evaluation under more conventional conditions, with a projection size increased to 8x8 and a moderate dropout rate of 0.25, showed a significant improvement. The model achieved a 95% accuracy on the MNIST test set, aligning well with standard benchmarks. This result highlights the effectiveness of the `SelfProjection` module when integrated into a neural network architecture under typical operating conditions.

### Insights
These evaluations indicate that the `SelfProjection` module is capable of extracting meaningful and robust features from the input data, even under stringent constraints. The improvement in performance with larger projection size and lower dropout rate suggests the module's potential in various application scenarios, especially in tasks requiring sophisticated data representation and processing.

Further experiments, including comparisons with baseline models and testing on more complex datasets, are planned to continue exploring the capabilities and optimizing the performance of the `SelfProjection` module.

## Contribution

Contributions to the `SelfProjection` module are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.
