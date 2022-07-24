import os

import gradio as gr
import torch
from gradio import SimpleCSVLogger
from torchvision import transforms

from data_loader import get_loader
from model import DecoderRNN, EncoderCNN
from nlp_utils import clean_sentence

cocoapi_dir = r"path/to/cocoapi/dir"

# # Defining a transform to pre-process the testing images.
transform_test = transforms.Compose(
    [
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

# Creating the data loader.
data_loader = get_loader(transform=transform_test, mode="test", cocoapi_loc=cocoapi_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the saved models to load.
encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


def predict_caption(image):
    if image is None:
        return "Please select an image"

    image = transform_test(image).unsqueeze(0)
    with torch.no_grad():
        # Moving image Pytorch Tensor to GPU if CUDA is available.
        image = image.to(device)

        # Obtaining the embedded image features.
        features = encoder(image).unsqueeze(1)

        # Passing the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features)

    sentence = clean_sentence(output, data_loader.dataset.vocab.idx2word)

    return sentence


gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=gr.Textbox(label="Predicted caption"),
    flagging_dir="./gradio_logs",
    flagging_callback=SimpleCSVLogger(),
).launch(share=True, server_port=7860)
