import json
import os

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from tqdm import tqdm

from vocabulary import Vocabulary

nltk.download("punkt")


class CoCoDataset(data.Dataset):
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        # create vocabulary from the captions
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        self.img_folder = img_folder
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")

            #  get list of tokens for each caption
            tokenized_captions = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]]["caption"]).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]

            # get len of each caption
            self.caption_lengths = [len(token) for token in tokenized_captions]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        # select random len
        sel_length = np.random.choice(self.caption_lengths)
        # find indices of captions having specific length
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        # select only limited (batch size) number of them
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train":
            return len(self.ids)
        else:
            return len(self.paths)


def get_loader(
    transform,
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    cocoapi_loc="/opt",
):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading.
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """

    assert mode in ["train", "test"], "mode must be one of 'train' or 'test'."

    if not vocab_from_file:
        assert (
            mode == "train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    elif mode == "train":
        if vocab_from_file:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, "cocoapi/images/train2014/")
        annotations_file = os.path.join(
            cocoapi_loc, "cocoapi/annotations/captions_train2014.json"
        )

    elif mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "cocoapi/images/test2014/")
        annotations_file = os.path.join(
            cocoapi_loc, "cocoapi/annotations/image_info_test2014.json"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # COCO caption dataset.
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
            ),
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader


if __name__ == "__main__":

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

    # Obtain the data loader.
    data_loader = get_loader(
        transform=transform_train,
        mode="train",
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_from_file=False,
        cocoapi_loc="",
    )
    print(data_loader)
