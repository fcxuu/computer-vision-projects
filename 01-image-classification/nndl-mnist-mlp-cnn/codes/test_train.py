# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
from mynn.data_augmentation import augment_dataset
from mynn.visualization import visualize_mlp_weights, visualize_model_training

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

# fixed seed for experiment
np.random.seed(309)

# Create output directories
if not os.path.exists('best_models'):
    os.makedirs('best_models')
if not os.path.exists('visualization'):
    os.makedirs('visualization')

# Load MNIST dataset
train_images_path = r'dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

# Apply data augmentation to training set (creating 1 additional example per original)
print("Applying data augmentation...")
augmented_train_imgs, augmented_train_labs = augment_dataset(train_imgs, train_labs, augmentation_factor=1)
print(f"Original training set: {train_imgs.shape[0]} images")
print(f"Augmented training set: {augmented_train_imgs.shape[0]} images")

# Create MLP model
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])

# Use MomentGD optimizer instead of SGD
optimizer = nn.optimizer.MomentGD(init_lr=0.01, model=linear_model, mu=0.9)

# Use ExponentialLR scheduler for smoother learning rate decay
scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

# Create loss function
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

# Create runner
runner = nn.runner.RunnerM(
    linear_model, 
    optimizer, 
    nn.metric.accuracy, 
    loss_fn, 
    batch_size=64,  # Increase batch size for better training
    scheduler=scheduler
)

# Train with improved settings
print("Training model...")
runner.train(
    [augmented_train_imgs, augmented_train_labs],  # Use augmented dataset 
    [valid_imgs, valid_labs], 
    num_epochs=5, 
    log_iters=100, 
    save_dir=r'best_models',
    eval_every=200  # Only evaluate validation set every 200 iterations
)

# Visualize training results
training_viz = visualize_model_training(runner)
training_viz.savefig('visualization/training_metrics.png')

# Visualize model weights
print("Visualizing model weights...")
# Visualize first layer weights (which should look like digit patterns)
weights_viz = visualize_mlp_weights(linear_model, layer_idx=0, figsize=(12, 12))
weights_viz.savefig('visualization/mlp_first_layer_weights.png')

# Plot training results with the built-in plot function
_, axes = plt.subplots(1, 2, figsize=(12, 5))
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.savefig('visualization/training_results.png')
plt.show()