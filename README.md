# SelfProjection Module for PyTorch
- [SelfProjection Module for PyTorch](#selfprojection-module-for-pytorch)
  - [Overview](#overview)
  - [Approach](#approach)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Evaluation](#evaluation)
    - [Experimental Setup and Results](#experimental-setup-and-results)
    - [Insights](#insights)
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

The `SelfProjection` module has been comprehensively evaluated using the MNIST dataset under various controlled conditions. These evaluations aim to assess its spatial feature extraction efficiency and overall model performance.

[eval_mnist.py](eval_mnist.py) - contains an evaluation code for MNIST dataset. By default it is set to extreme conditions with heavy projection reduction (4x4), high dropout rate at projection level (0.75) and extreme dropout rate at input level (0.9).

### Experimental Setup and Results

Three distinct experimental setups were employed, each with specific configurations of projection size and dropout rates:

1. **Standard Conditions**:
   - **Configuration**: Projection size of 8x8, input dropout rate of 0.0, and projection dropout rate of 0.25.
   - **Command**:
    ```bash
    python eval_mnist.py --seed=1 --p-size=8 --dropout-rate-i=0.0 --dropout-rate-p=0.25 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
    ```
   - **Results**: Achieved an average loss of 0.1484 and an accuracy of 95% (9544/10000) on the test set.
   - **Analysis**: This performance under standard conditions demonstrates the module's effectiveness in capturing essential features necessary for high accuracy in digit classification.

2. **Heavy Reduction with High Dropout**:
   - **Configuration**: Projection size of 4x4, input dropout rate of 0.0, and projection dropout rate of 0.75.
   - **Command**:
    ```bash
    python eval_mnist.py --seed=1 --p-size=4 --dropout-rate-i=0.0 --dropout-rate-p=0.75 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
    ```
   - **Results**: Achieved an average loss of 0.7027 and an accuracy of 79% (7919/10000) on the test set.
   - **Analysis**: The model shows resilience and robust feature extraction capability even with substantial dimensionality reduction and high dropout, though with a noticeable decrease in accuracy compared to the standard setup.

3. **Heavy Reduction with High Dropout of Projection and Extreme Dropout of Input**:
   - **Configuration**: Projection size of 4x4, input dropout rate of 0.9, and projection dropout rate of 0.75.
   - **Command**:
    ```bash
    python eval_mnist.py --seed=1 --p-size=4 --dropout-rate-i=0.9 --dropout-rate-p=0.75 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
    ```
   - **Results**: Achieved an average loss of 1.1053 and an accuracy of 69% (6868/10000) on the test set.
   - **Analysis**: Under these extreme conditions, the model still performs significantly above chance level, indicating the `SelfProjection` module's ability to extract meaningful features from highly sparse data, albeit with a reduction in overall performance.


### Insights

These varied evaluations illustrate the `SelfProjection` module's capability to adapt and extract meaningful features across a range of scenarios. The results under different projection sizes and dropout rates provide valuable insights into the module's potential applicability in tasks requiring nuanced data representation and processing. The module demonstrates robustness in feature extraction, especially notable under the heavy reduction and high dropout conditions.

### Future Directions

The ongoing development of the `SelfProjection` module is a component of a personal endeavor in the field of machine learning. Future plans include conducting experiments with more complex datasets to further assess and refine the module's capabilities. These steps are aimed at exploring a broader range of applications and enhancing the module's performance in diverse settings.

This module is a reflection of my interest in contributing to the machine learning community through individual efforts and is one aspect of a larger personal project dedicated to exploring innovative approaches in the field.

## Contribution

Contributions to the `SelfProjection` module are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.
