import torch
import torch.nn as nn
from torchvision.transforms import v2
import os



device = "cuda" if torch.cuda.is_available() else "cpu"

img_preprocessing = v2.Compose([
        v2.Resize((150, 150)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

def load_image(img_no, transformation):
    # define the path
    # path = "/home/skandan-c-y/PyCharmProjects/PyTorch"\
    #        f"/data/intel-image-classification/seg_pred/seg_pred/{img_no}.jpg"
    #
    path = '/home/skandan-c-y/PycharmProjects/PyTorch/data/intel-image-classification/seg_pred/seg_pred'\
            f'/{img_no}.jpg'

    if not os.path.exists(path):
        raise FileNotFoundError("Image not found...")

    # load the image
    img = Image.open(path).convert('RGB')

    image = transformation(img)
    return image.unsqueeze(0).to(device)


def preprocess_image(image, transformation):
    image = image.convert("RGB")
    image = transformation(image)
    return image.unsqueeze(0).to(device)


def predict(model, img):
    CLASS_NAMES = ["buildings", "forest", "glacier",
                   "mountain", "sea", "street"
    ]

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[prediction.item()]
    conf = confidence.item() * 100

    return predicted_class, conf



# define the  model

class CNN4(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(          # (1, 150, 150): (3, 150, 150)
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # (32, 150, 150): (32, 75, 75)
            nn.Conv2d(            # (32, 75, 75): (64, 75, 75)
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # (64, 75, 75): (64, 37, 37)
            nn.Conv2d(           # (64, 37, 37): (128, 37, 37(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                    # (128, 37, 37): (128, 18, 18)
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*18*18, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.clf(x)
        return x


cnn = CNN4()
cnn.load_state_dict(
    torch.load("models/torch04_cnn4_state_dict.pth",
               map_location=device,
                weights_only=True)
)
cnn.to(device)
cnn.eval()



# serving using fastapi

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    tensor = preprocess_image(pil_image, img_preprocessing)
    predicted_class, confidence = predict(cnn, tensor)

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }

# successful
