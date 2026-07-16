# Animal Species Image Classification using EfficientNet-B0 Transfer Learning

Classifying animal species from images using transfer learning with EfficientNet-B0 and PyTorch.

**Live API:** [Link](https://animal-species-transfer-learning.onrender.com/docs)

---

## Overview

This project performs 10-class animal species image classification using transfer learning.

Instead of training a convolutional neural network from scratch, a pretrained EfficientNet-B0 model is used as a feature extractor while a custom classification head is trained on the target dataset. The project also includes an inference pipeline, FastAPI REST API, and Docker containerization for deployment.

---

## Model Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Framework:** PyTorch
- **Transfer Learning Strategy:** Frozen feature extractor
- **Classification Head:**
  - Dropout (0.2)
  - Linear (1280 → 10)
- **Input Resolution:** 224 × 224 RGB images
- **Image Preprocessing:** `EfficientNet_B0_Weights.DEFAULT.transforms()`

---

## Training Configuration

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10
- **Train / Validation Split:** 80 / 20
- **Checkpointing:** Best model saved based on validation accuracy

---

## Performance

**Best Validation Accuracy:** **96.05%**

The model achieved 96.05% validation accuracy while training only the custom classification head, demonstrating the effectiveness of transfer learning for image classification.

---

## Deployment

The project includes:

- FastAPI inference API
- Docker support for reproducible deployment.
- Image upload endpoint
- Confidence score prediction
- Saved model checkpoint loading

---

## Dataset

This project uses the Animals-10 dataset containing images from 10 animal species. The dataset is not included in this repository and should be placed in the `data/` directory before retraining the model.

The pretrained checkpoint (`models/best_model.pth`) is included, so the API can be used immediately without retraining.

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/vanshikacy/animal-species-transfer-learning
cd animal-species-transfer-learning
```

### Build the Docker image

```bash
docker build -t animal-classification-api:1.0 .
```

### Run the container

```bash
docker run -p 8000:8000 animal-classification-api:1.0
```

### Access the API

Swagger UI:

```
http://localhost:8000/docs
```

Interactive API documentation:

```
http://localhost:8000/redoc
```

---

## API Endpoint

### Predict Animal Species

**POST**

```
/predict
```

Upload an image using multipart/form-data.

**Response**

```json
{
  "prediction": {
    "class_id": 9,
    "class_name": "squirrel",
    "confidence": 99.98
  }
}
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- FastAPI
- Docker
- Pillow

The model is served through a FastAPI REST API and can be deployed locally using Docker.

