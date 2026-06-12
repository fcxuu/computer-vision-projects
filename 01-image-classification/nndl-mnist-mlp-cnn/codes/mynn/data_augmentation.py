import numpy as np
from scipy.ndimage import rotate, shift, zoom
import random

def augment_image(image, label, rotate_range=15, shift_range=2, zoom_range=(0.9, 1.1)):
    """
    Apply random transformations to an image for data augmentation
    
    Args:
        image: input image (28x28 for MNIST)
        label: corresponding label
        rotate_range: maximum rotation angle in degrees
        shift_range: maximum shift in pixels
        zoom_range: range of zoom factors
    
    Returns:
        transformed image and original label
    """
    # Make a copy of the image to avoid modifying the original
    img = image.copy()
    
    # Random rotation
    if rotate_range > 0:
        angle = random.uniform(-rotate_range, rotate_range)
        img = rotate(img, angle, reshape=False, mode='nearest')
    
    # Random shift
    if shift_range > 0:
        dx = random.uniform(-shift_range, shift_range)
        dy = random.uniform(-shift_range, shift_range)
        img = shift(img, [dy, dx], mode='nearest')
    
    # Random zoom
    if zoom_range[0] < zoom_range[1]:
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        # Get current shape
        shape = img.shape
        # Zoom the image
        img = zoom(img, zoom_factor, mode='nearest')
        # Resize back to original size
        if zoom_factor > 1:  # If zoomed in, crop center
            new_shape = img.shape
            start_y = (new_shape[0] - shape[0]) // 2
            start_x = (new_shape[1] - shape[1]) // 2
            img = img[start_y:start_y+shape[0], start_x:start_x+shape[1]]
        else:  # If zoomed out, pad with zeros
            new_shape = img.shape
            pad_y = (shape[0] - new_shape[0]) // 2
            pad_x = (shape[1] - new_shape[1]) // 2
            # Create a new array of zeros with the original shape
            new_img = np.zeros(shape)
            # Place the zoomed image in the center
            new_img[pad_y:pad_y+new_shape[0], pad_x:pad_x+new_shape[1]] = img
            img = new_img
    
    return img, label

def augment_dataset(images, labels, augmentation_factor=1):
    """
    Augment a dataset by creating additional samples
    
    Args:
        images: original images
        labels: original labels
        augmentation_factor: how many additional samples to create per original sample
    
    Returns:
        augmented dataset (images and labels)
    """
    # Get the original shape of the images
    if len(images.shape) == 4:  # [batch, channels, height, width]
        N, C, H, W = images.shape
        images_reshaped = images.transpose(0, 2, 3, 1).reshape(N, H, W, C)
        is_4d = True
    else:  # [batch, height*width]
        N = images.shape[0]
        H = W = int(np.sqrt(images.shape[1]))
        images_reshaped = images.reshape(N, H, W)
        is_4d = False
    
    augmented_images = []
    augmented_labels = []
    
    # Include original data
    augmented_images.append(images)
    augmented_labels.append(labels)
    
    # Create augmented samples
    for i in range(augmentation_factor):
        aug_batch_images = []
        for j in range(N):
            if is_4d:
                # For CNN input format [batch, channels, height, width]
                aug_img = np.zeros_like(images_reshaped[j])
                for c in range(C):
                    aug_img[:,:,c], _ = augment_image(images_reshaped[j,:,:,c], labels[j])
                aug_batch_images.append(aug_img.transpose(2, 0, 1))  # Back to [channels, height, width]
            else:
                # For MLP input format [batch, height*width]
                aug_img, _ = augment_image(images_reshaped[j], labels[j])
                aug_batch_images.append(aug_img.flatten())
        
        if is_4d:
            augmented_images.append(np.array(aug_batch_images))
        else:
            augmented_images.append(np.array(aug_batch_images))
        augmented_labels.append(labels)
    
    # Concatenate all augmented data
    augmented_images = np.vstack(augmented_images)
    augmented_labels = np.hstack(augmented_labels)
    
    return augmented_images, augmented_labels