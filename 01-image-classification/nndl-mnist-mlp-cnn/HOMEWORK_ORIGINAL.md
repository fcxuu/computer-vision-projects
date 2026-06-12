# Neural Network and Deep Learning - Project 1
**MNIST Handwritten Digit Classification**

## Abstract

This project implements a neural network framework from scratch using NumPy for handwritten digit classification on the MNIST dataset. The main components include multi-layer perceptron (MLP), convolutional neural network (CNN), various optimizers, loss functions, and regularization techniques. The implementation focuses on educational purposes, building the essential components of a neural network without relying on deep learning frameworks.

## 1. Project Structure

```
├── mynn/                  # Neural network implementation
│   ├── op.py              # Basic operators (Linear, Conv2D, ReLU, Sigmoid)
│   ├── optimizer.py       # Optimizers (SGD, MomentGD)
│   ├── lr_scheduler.py    # Learning rate schedulers
│   ├── models.py          # Model definitions (MLP, CNN)
│   ├── runner.py          # Training utilities
│   └── metric.py          # Evaluation metrics
├── test_train.py          # MLP model training script
└── test_model.py          # Model testing script
```

## 2. Implementation Details

### 2.1 Network Structures

#### Multi-Layer Perceptron (MLP)
- Implemented a flexible MLP architecture with configurable layer sizes
- Supported various activation functions (ReLU, Sigmoid)
- Included weight decay regularization
- Implemented forward and backward propagation

#### Convolutional Neural Network (CNN)
- Implemented 2D convolutional layers with configurable parameters
- Supported kernel size, stride, and padding options
- Combined convolutional layers with fully connected layers
- Implemented proper reshaping between convolutional and fully connected layers

### 2.2 Training Procedures

#### Optimizers
- **SGD (Stochastic Gradient Descent)**: Basic implementation with learning rate
- **MomentGD (Momentum Gradient Descent)**: Enhanced SGD with momentum term using:
  ```
  w_{t+1} = w_t - α_t∇f(w_t) + β_t(w_t - w_{t-1})
  ```
  where α_t is the learning rate and β_t is the momentum strength (typically 0.9)

#### Learning Rate Schedulers
- **StepLR**: Reduces learning rate by a factor at specified epochs
- **MultiStepLR**: Reduces learning rate at multiple milestones
- **ExponentialLR**: Applies exponential decay to learning rate

### 2.3 Regularization Methods

- **L2 Regularization (Weight Decay)**: Implemented in Linear and Conv2D layers
  - Adds a penalty term to the loss function: λ‖w‖²/2
  - Helps prevent overfitting by discouraging large weights
- **Early Stopping**: Implemented in the training runner
  - Saves the best model based on validation accuracy
  - Prevents overfitting by stopping training when validation metrics plateau

### 2.4 Loss Functions

- **MultiCrossEntropyLoss**: Implemented for multi-class classification
  - Incorporated softmax function to convert logits to probabilities:
    ```
    p(y_i) = exp(z_i) / ∑_j exp(z_j)
    ```
  - Computed cross-entropy loss between predictions and targets
  - Implemented efficient backpropagation for softmax + cross-entropy

### 2.5 Data Augmentation

Implemented several augmentation techniques to create additional training examples:
- Random rotations (±15 degrees)
- Random shifts (±2 pixels)
- Random zooming (0.9-1.1x)

### 2.6 Visualization

Implemented tools to visualize:
- Model training metrics (loss and accuracy)
- MLP layer weights
- CNN convolutional kernels
- Model predictions on test samples

## 3. Experimental Results

### 3.1 MLP Results

The MLP model with the following configuration:
- Architecture: 784 (input) → 600 (hidden) → 10 (output)
- Activation: ReLU
- Optimizer: MomentGD (lr=0.01, momentum=0.9)
- Scheduler: ExponentialLR (gamma=0.999)
- Weight decay: 1e-4

Achieved:
- Training accuracy: ~98%
- Test accuracy: ~93%

### 3.2 CNN Results

The CNN model with the following configuration:
- Convolutional layers: 1 → 16 → 32 channels, 5×5 kernels
- Fully connected layers: N → 100 → 10 (where N depends on conv output)
- Activation: ReLU
- Optimizer: MomentGD (lr=0.01, momentum=0.9)
- Scheduler: MultiStepLR (milestones=[400, 800, 1200], gamma=0.5)
- Weight decay: 1e-5

Achieved:
- Training accuracy: ~99%
- Test accuracy: ~98%

### 3.3 Effect of Data Augmentation

Data augmentation significantly improved model robustness:
- Without augmentation: ~92% test accuracy (MLP)
- With augmentation: ~93% test accuracy (MLP)

The improvement was more noticeable in scenarios with less training data, demonstrating the value of augmentation in expanding the effective dataset size.

### 3.4 Weight Visualization

Visualizing the first layer weights of the MLP model revealed interpretable patterns resembling digit fragments, confirming that the model learned meaningful features. The CNN kernels in the first layer showed edge detectors and texture filters, progressing to more complex feature detectors in deeper layers.

## 4. Discussion and Future Work

### Architectural Choices

The experiments confirmed that CNNs outperform MLPs on image classification tasks due to their ability to capture spatial hierarchies. The significant accuracy improvement (93% → 98%) demonstrates the importance of architectural choices for specific data types.

### Optimization Strategy

Momentum-based optimization consistently outperformed vanilla SGD, converging faster and achieving better final accuracy. The learning rate scheduler was crucial for reaching optimal performance, preventing both premature convergence and instability.

### Regularization Effectiveness

Weight decay proved essential for good generalization, especially for the MLP model. Finding the optimal regularization strength required experimentation, with 1e-4 providing the best balance between underfitting and overfitting.

### Future Improvements

Potential enhancements to explore:
- Batch normalization for faster and more stable training
- Dropout for improved regularization
- More advanced architectures like ResNet blocks
- Additional data augmentation techniques (elastic distortions, cutout)

## 5. Conclusion

This project successfully implemented a neural network framework from scratch, demonstrating the fundamental concepts of deep learning without relying on existing frameworks. The implementation achieved competitive results on the MNIST dataset, with the CNN model reaching 98% accuracy, comparable to many published benchmarks.

The exploration of different architectures, optimization methods, and regularization techniques provided valuable insights into the practical aspects of neural network design and training. The results confirmed several theoretical principles, such as the superiority of CNNs for image data and the benefits of momentum-based optimization.

### Training MLP Model

```bash
python test_train.py
```

### Testing MLP Model

```bash
python test_model.py
```





