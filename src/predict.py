import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

weights=EfficientNet_B0_Weights.DEFAULT

model=efficientnet_b0(weights=weights)

class_names=[
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "spider",
    "squirrel",
]

num_classes=len(class_names)

model.classifier[1]=nn.Linear(
    in_features=model.classifier[1].in_features,
    out_features=num_classes
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

loaded_weights=torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(loaded_weights) 

# old way-
# image=Image.open(
#     r"data\test-img\African_Bush_Elephant.jpg.webp"
# ).convert("RGB")
# image=weights.transforms()(image)
# image=image.unsqueeze(0)
# images=image.to(device) 
# model.eval()
# with torch.no_grad():
#     outputs=model(image) 
#     _, predicted=torch.max(outputs, dim=1)
#     predicted_class=class_names[predicted.item()]
#     probabilities=torch.softmax(outputs, dim=1)
#     confidence=probabilities[0][predicted.item()]*100
#     print(f"predicted class: {predicted_class}")
#     print(f"Confidence: {confidence:.2f}%")

# new way-
def predict_image(ip_image): # api will handle image.open
    image=weights.transforms()(ip_image)
    image=image.unsqueeze(0)
    image_tensor=image.to(device)

    model.eval()

    with torch.no_grad():
        outputs=model(image_tensor)
        _, predicted=torch.max(outputs, dim=1)

        predicted_class=class_names[predicted.item()]

        probabilities=torch.softmax(outputs, dim=1)
        confidence=probabilities.max().item()
    
    return {
        "class_name": predicted_class,
        "confidence": round(confidence, 4)
    }

    



