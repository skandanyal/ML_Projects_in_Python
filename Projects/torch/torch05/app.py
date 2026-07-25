import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.models import resnet18
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from pathlib import Path



device = "cuda" if torch.cuda.is_available() else "cpu"

img_preprocessing_cnn = v2.Compose([
        v2.Resize((150, 150)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

img_preprocessing_rn = v2.Compose([
        v2.Resize((224, 224)),
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
    path = '/torch05/data/intel-image-classification/seg_pred/seg_pred' \
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

class CNN_skip_connn(nn.Module):
    def __init__(self):
        super().__init__()

        # conv block 1
        # (3, 150, 150): (32, 75, 75)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv block 2a
        # (32, 75, 75): (64, 75, 75)
        self.conv2ai = nn.Conv2d(32, 64, padding=1, kernel_size=3)
        self.relu2ai = nn.ReLU()
        # (64, 75): (64, 37, 37)
        self.conv2aii = nn.Conv2d(64, 64, padding=1, kernel_size=3)

        # conv block 2b
        # (32, 75, 75): (64, 37, 37)
        self.conv2b = nn.Conv2d(32, 64, kernel_size=1)

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv block 3
        # (64, 37, 37): (128, 18, 18)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*18*18, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        # conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # residual block
        identity = self.conv2b(x)

        out = self.conv2ai(x)
        out = self.relu2ai(out)

        out = self.conv2aii(out)

        # connecting the skip conn.n.
        out = out + identity


        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        x = self.clf(out)

        return x


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

cnn_path = MODEL_DIR / "torch05_cnn_skip_state_dict.pth"
rn_path = MODEL_DIR / "torch05_rn_state_dict.pth"


cnn = CNN_skip_connn()
cnn.load_state_dict(
    torch.load(cnn_path,
               map_location=device,
                weights_only=True)
)
cnn.to(device)
cnn.eval()


rn = resnet18(weights=None)
rn.fc = nn.Linear(rn.fc.in_features, 6)
rn.load_state_dict(
    torch.load(rn_path,
               map_location=device,
               weights_only=True)
)
rn = rn.to(device)
rn.eval()


# serving using fastapi

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

app = FastAPI(title="Intel Image Classifier")

# Serve static assets (the frontend lives in torch05/static/)
_here = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_here / "static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    return FileResponse(_here / "static" / "index.html")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return {'error': 'Invalid file format'}

    tensor_cnn = preprocess_image(pil_image, img_preprocessing_cnn)
    tensor_rn  = preprocess_image(pil_image, img_preprocessing_rn)

    prediction_cnn, confidence_cnn = predict(cnn, tensor_cnn)
    prediction_rn,  confidence_rn  = predict(rn,  tensor_rn)

    return {
        'cnn': {
            'prediction': prediction_cnn,
            'confidence': round(confidence_cnn, 4)
        },
        'rn': {
            'prediction': prediction_rn,
            'confidence': round(confidence_rn, 4)
        }
    }


# run with:  uvicorn trial05:app --reload --port 8000
