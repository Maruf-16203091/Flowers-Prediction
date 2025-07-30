

# 🌼 Flower Classification Using CNN (PyTorch)

This project demonstrates how to build and train a simple **Convolutional Neural Network (CNN)** using **PyTorch** to classify flower images into five categories: **daisy**, **dandelion**, **rose**, **sunflower**, and **tulip**.

---

## 📌 Features

* ✅ Image classification with CNN (from scratch)
* ✅ Custom dataset using `ImageFolder`
* ✅ Image transformations: resize, normalize
* ✅ GPU support via CUDA
* ✅ Evaluation with accuracy

---

## 🧠 Model Architecture

```
Input Image (64x64x3)
   ↓
Conv2D (3→16), ReLU, MaxPool → 64x64x16 → 32x32x16
   ↓
Conv2D (16→32), ReLU, MaxPool → 32x32x32 → 16x16x32
   ↓
Flatten → Linear → ReLU → Linear → Output
```

---

## 🗂️ Dataset

* 📁 Folder structure should follow:

```
flowers/
  daisy/
  dandelion/
  rose/
  sunflower/
  tulip/
```

* 🔄 Loaded using `ImageFolder`, then split into 80% training and 20% test data.

---

## ⚙️ Requirements

* Python 3.x
* PyTorch
* torchvision
* Google Colab (for Drive access, GPU)

---

## 🚀 How to Run

1. **Mount Google Drive** and load dataset:

```python
from google.colab import drive
drive.mount('/content/drive')

dataset = ImageFolder(root='/content/drive/MyDrive/ML_Datasets/flowers', transform=transform)
```

2. **Preprocess Images**:

```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
```

3. **Build the CNN Model**:

```python
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

4. **Train the Model**:

```python
for epoch in range(num_epochs):
    ...
    print(f"Epoch {epoch+1}, Loss: ..., Accuracy: ...")
```

---

## 📊 Results

| Epoch | Accuracy |
| ----- | -------- |
| 1     | 46.88%   |
| 5     | 82.59%   |
| 10    | 98.97%   |

✅ Achieved nearly **99% accuracy** on the training dataset after 10 epochs.

---



