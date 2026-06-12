import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from mynn.visualization import visualize_mlp_weights
import os

# 确保目录存在
if not os.path.exists('visualization'):
    os.makedirs('visualization')

# 修改模型路径
model_path = r'best_models/best_model.pickle'
if not os.path.exists(model_path):
    print(f"Warning: Model file not found at {model_path}")
    print("Make sure you have run test_train.py first to train the model")
    model_path = input("Enter the correct model path (or press Enter to exit): ")
    if not model_path:
        print("Exiting...")
        exit()

# 加载模型
model = nn.models.Model_MLP()
model.load_model(model_path)

# 加载测试数据
test_images_path = r'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 正确的归一化方式
test_imgs = test_imgs / 255.0

# 计算测试集准确率
logits = model(test_imgs)
test_accuracy = nn.metric.accuracy(logits, test_labs)
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 可视化模型权重（第一层应该显示数字模式）
print("Visualizing model weights...")
weights_viz = visualize_mlp_weights(model, layer_idx=0, figsize=(12, 12))
weights_viz.savefig('visualization/test_mlp_weights.png')
plt.close()

# 随机选择一些测试样本进行预测和可视化
num_samples = 10
sample_indices = np.random.choice(len(test_imgs), num_samples, replace=False)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    img = test_imgs[idx].reshape(28, 28)
    label = test_labs[idx]
    
    # 获取模型预测
    prediction = np.argmax(model(test_imgs[idx:idx+1]))
    
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    
    if prediction == label:
        plt.title(f"Pred: {prediction}, True: {label}", color='green')
    else:
        plt.title(f"Pred: {prediction}, True: {label}", color='red')
    
    plt.axis('off')

plt.suptitle("Model Predictions on Test Samples")
plt.tight_layout()
plt.savefig('visualization/test_predictions.png')
plt.show()