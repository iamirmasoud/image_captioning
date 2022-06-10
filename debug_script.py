import torch
from torchvision import transforms

from data_loader import get_loader
from model import DecoderRNN, EncoderCNN

# Define a transform to pre-process the training images.
transform_train = transforms.Compose(
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

# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

cocoapi_dir = r"/media/masoud/F60C689F0C685C9D/immediate D/Course_Assignments/FINISHED/VISION/Udacity - " \
              r"Computer Vision Nanodegree/PROJECTS/2 - IMAGE CAPTIONING/MY/"

# Obtain the data loader (from file). Note that it runs much faster than before!
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True,
                         cocoapi_loc=cocoapi_dir)

# # Obtain the batch.
# images, captions = next(iter(data_loader))
#
# print('images.shape:', images.shape)
# print('captions.shape:', captions.shape)
#
# torch.save(images, 'images.pt')
# torch.save(captions, 'captions.pt')

# Load tensors
images, captions = torch.load("images.pt"), torch.load("captions.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the dimensionality of the image embedding.
image_embed_size = 256

# Initialize the encoder.
encoder = EncoderCNN(image_embed_size)

# Move the encoder to GPU if CUDA is available.
encoder.to(device)

# # Move last batch of images (from Step 2) to GPU if CUDA is available.
images = images.to(device)


# Pass the images through the encoder.
features = encoder(images)

print("type(features):", type(features))
print("features.shape:", features.shape)
print("captions.shape:", captions.shape)

# Check that the encoder satisfies some requirements of the project!
assert type(features) == torch.Tensor, "Encoder output needs to be a PyTorch Tensor."

assert (features.shape[0] == batch_size) and (
    features.shape[1] == image_embed_size
), "The shape of the encoder output is incorrect."

# Specify the number of features in the hidden state of the RNN decoder.
hidden_size = 512
word_embed_size = image_embed_size
# Store the size of the vocabulary.
# vocab_size = 8855
vocab_size = len(data_loader.dataset.vocab)

# Initialize the decoder.
decoder = DecoderRNN(word_embed_size, hidden_size, vocab_size)

# Move the decoder to GPU if CUDA is available.
decoder.to(device)

# Move last batch of captions (from Step 1) to GPU if CUDA is available
captions = captions.to(device)

# Pass the encoder output and captions through the decoder. (bs, cap_len, vocab_size)
outputs = decoder(features, captions)

print("type(outputs):", type(outputs))
print("outputs.shape:", outputs.shape)

# Check that the decoder satisfies some requirements of the project!
assert type(outputs) == torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
assert (
    (outputs.shape[0] == batch_size)
    and (outputs.shape[1] == captions.shape[1])
    and (outputs.shape[2] == vocab_size)
), "The shape of the decoder output is incorrect."
