import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ----- Model definition (must match training) -----
class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()

        # feature extractor
        self.cnn_nw = nn.Sequential(
            nn.Conv2d(in_channels=1,                # shape: ip:28, op:floor(28+2*0-3/1)+1=26
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),         # shape: ip:13, op:((26-2*1+2)/2)+1=13
                         stride=(2,2),
                         padding=1),

            nn.Conv2d(in_channels=32,               # shape: ip:13, op:floor((13-3)/1)+1=11
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),         # shape: ip:11, op:((11-2)/2)+1=6
                         stride=(2,2))
        )

        # network
        self.clf = nn.Sequential(
            nn.Flatten(),                           # shape: 64*1*1 = 64
            nn.Linear(in_features=64*6*6, out_features=128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.cnn_nw(x)
        x = self.clf(x)
        return x


# ----- Load model -----
device = torch.device("cpu")
model = cnn_model().to(device)
model.load_state_dict(torch.load("practice/mnist_cnn.pth", map_location=device))
model.eval()

# ----- UI -----
st.title("MNIST Digit Classifier")

canvas = st_canvas(
    fill_color="black",
    stroke_width=8,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ----- Prediction -----
if canvas.image_data is not None:
    img = canvas.image_data.astype("uint8")

    # convert to PIL
    img = Image.fromarray(img)
    img = img.convert("L")              # grayscale
    img = ImageOps.invert(img)          # invert (white digit on black)

    # resize to 28x28
    img = img.resize((28, 28))

    # normalize
    img = np.array(img) / 255.0

    # to tensor
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    st.write(f"MLP Prediction: {pred}")
    st.write(f"CNN Prediction: {pred}")