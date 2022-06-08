import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from googletrans import Translator
from PIL import Image
from torchvision import transforms

from model import DecoderRNN, EncoderCNN
from vocabulary import Vocabulary

vocab = Vocabulary(vocab_threshold=5, vocab_from_file=True)

# vocab = data_loader.dataset.vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the saved models to load.
encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


def clean_sentence(output, vocab):
    sentence = ""
    for i in output:
        word = vocab.idx2word[i]
        if i == 0:
            continue
        elif i == 1:
            break
        if i == 18:  # comma
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence.strip()


def get_prediction(img_path, vocab, encoder, decoder):
    img_pil = Image.open(img_path)

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

    img_tensor = transform_test(img_pil)

    plt.imshow(np.squeeze(img_tensor.numpy()).transpose(1, 2, 0))
    plt.title("example image")
    plt.show()

    image = img_tensor.to(device)
    image.unsqueeze_(0)
    # Obtain the embedded image features.
    features = encoder(image).unsqueeze(1)

    # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)

    sentence = clean_sentence(output, vocab)
    return sentence


generated_caption = get_prediction("bride.jpg", vocab, encoder, decoder)
print(generated_caption)
translator = Translator()

try:
    translator = Translator()
    print(translator.translate(generated_caption, src="en", dest="fa").text)
except:
    print("Cannot connect to google translate.")
