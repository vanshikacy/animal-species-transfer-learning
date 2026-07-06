import torch 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch.nn as nn

weights=EfficientNet_B0_Weights.DEFAULT

train_transforms=weights.transforms()

dataset=datasets.ImageFolder(
    root="data/raw-img",
    transform=train_transforms
)

train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size

train_dataset, val_dataset=random_split(
    dataset, 
    [train_size, val_size]
)

batch_size=32

train_loader=DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader=DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

model=efficientnet_b0(weights=weights)

# print(model.classifier) 
# print(model.classifier[1])

model.classifier[1]=nn.Linear(
    in_features=model.classifier[1].in_features, 
    out_features=len(dataset.classes)
)

# print(model)

for param in model.features.parameters():
    param.requires_grad=False

# print(model.features[0][0].weight.requires_grad)
# print(model.classifier[1].weight.requires_grad) 
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device) 

criterion=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


epochs=10
best_accuracy=0.0

for epoch in range(epochs):

    #training 
    model.train()
    running_loss=0.0

    for images, labels in train_loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    #validation
    model.eval()
    val_loss=0.0
    correct=0
    total=0
        
    with torch.no_grad():
        for images, labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)
            
            outputs=model(images)
            loss=criterion(outputs, labels)
            val_loss+=loss.item()
                
            _, predicted=torch.max(outputs, dim=1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0) 

    #train loss
    epoch_loss=running_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

    #validation metrics
    val_loss/=len(val_loader)
    accuracy=100*correct/total
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    #saving the best model 
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Best model saved!")





       

        

        
        






