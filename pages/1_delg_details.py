import streamlit as st

# Custom Styling
st.markdown("""
    <style>
        body {
            font-family: 
        }
        /* Center content and limit max width */
            .block-container {
            padding-left: 5rem;
            padding-right: 5rem;
            max-width: 80% !important;
        }

        /* Reduce sidebar width for better responsiveness */
        [data-testid="stSidebar"] {
            width: 250px;
        }

        /* Hide sidebar toggle button when sidebar is collapsed */
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                display: none;
            }
        }
            
        .image-caption {
            font-size: 14px;
            font-style: italic;
            color: gray;
        }

        /* Center align headers */
        .h1 {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        }

        .h2 {
        font-size: 36px;
        padding-top: 30px;
        }

        .h3 {
        font-size: 24px;
        padding-top: 10px;
        }
            
        .custom-markdown tag{
            color: #FE4A4B;
        }

    </style>
""", unsafe_allow_html=True)
st.markdown ('<div class="h1">Deep Local and Global Image Features (DELG) </div>', unsafe_allow_html=True)
st.write("""
         deep local and global image feature methods, which are particularly useful for the computer vision tasks 
            of instance-level recognition and retrieval.
            These were introduced in the [DELF](https://arxiv.org/abs/1612.06321), 
            [Detect-to-Retrieve](https://arxiv.org/abs/1812.01584), [DELG](https://arxiv.org/abs/2001.05027) and Google Landmarks Dataset v2 papers.
""")

st.markdown('<div class="h2">Dataset Overview</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
st.write("""    
    By the given data of this project. This dataset is a collection of images of all 18 locations.
""")

col1, col2 = st.columns(2)
with col1:
    st.image("asset/dataset_structure.png", width=300)
    
with col2: 
    st.write("""
        The dataset contains 18 folders, which is test, train and val. In folder train have a separated folder for 
        each class (location).  Each folder contains images of the respective class. 
        The dataset is structured as follows:
    """)
    st.markdown("""
        <div style="font-size:16px; line-height:1.6;"> 
        <ul>
            <li>Train Images <span style="float:right;">2,865 &nbsp; Files (.jpg)</span></li>
            <li>Test Images <span style="float:right;">18 &nbsp; Files (.jpg)</span></li>
            <li>Validate Images <span style="float:right;">159 &nbsp; Files (.jpg)</span></li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="h2">Preprocessing images</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True) 
    
st.code("""
    import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

def scan_dataset(root_dir):
    
    #root_dir: 'data/train' or 'data/val' or 'data/test'
    #Returns: list of image paths and their class names
    
    image_paths = []
    labels = []
    
    for class_name in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in sorted(os.listdir(class_folder)):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                image_paths.append(os.path.join(class_folder, fname))
                labels.append(class_name)
    
    return image_paths, labels

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None): # Stores image paths, labels, and transforms.
        self.image_paths = image_paths
        self.labels = labels  # None for test set
        self.transform = transform
        
    def __len__(self): # returns the number of images in the dataset.
        return len(self.image_paths)
    
    def __getitem__(self, idx): # loads and returns an image and its label (if available).
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img, img_path  # return path for query images

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

train_paths, train_labels = scan_dataset("data/train")
val_paths, val_labels = scan_dataset("data/val")
test_paths, _ = scan_dataset("data/test")

classes = sorted(set(train_labels))
class_to_idx = {c:i for i,c in enumerate(classes)}
train_labels_idx = [class_to_idx[c] for c in train_labels]
val_labels_idx = [class_to_idx[c] for c in val_labels]

train_dataset = LandmarkDataset(train_paths, train_labels_idx, train_transform)
val_dataset = LandmarkDataset(val_paths, val_labels_idx, val_transform)
test_dataset = LandmarkDataset(test_paths, labels=None, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


""")

st.markdown('<div class="h2">Training loop</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True) 

st.code("""
        # Classification head for fine-tuning global features
classifier = nn.Linear(2048, num_classes).to(device)

# Loss and optimizer (only optimize global + classifier)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.backbone.parameters()) + list(model.gem.parameters()) + list(classifier.parameters()),
    lr=learning_rate
)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(num_epochs):
    model.train()
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        feats = model(imgs)['global']           # only global descriptors
        outputs = classifier(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    classifier.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model(imgs)['global']
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100.*val_correct/val_total
    val_loss_avg = val_loss / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Validation Loss: {val_loss_avg:.4f} | Validation Accuracy: {val_acc:.2f}%\n")
""")