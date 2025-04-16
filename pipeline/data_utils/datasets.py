import os
import torch
import shutil
import warnings
import logging
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import IterableDataset, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, CelebA, Places365

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = [
    "CIFAR10",
    "CIFAR100",
    "CelebA",
    "Places365",
    "ImageNet",
    "CC3M",
    "Flickr8k",
    "CUB",
    "CUBA",
]

CAPTION_DATASETS = [
    "CIFAR10",
    "CIFAR100",
    "Places365",
    "ImageNet",
    
]

MULTICAPTION_DATASETS = [
    "CelebA",
    "CUBA",
]

IMG2TXT_DATASETS = [
    "CC3M",
    "Flickr8k",
    "CUB"
]

SUPPORTED_VOCAB = [
    "laion",
    "laion_unigram",
    "laion_bigrams",
    "mscoco"
]
class VocabDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for item in self.data:
            yield None, item
    
    def __len__(self):
        return len(self.data)

class CelebAMy(IterableDataset):
    def __init__(self, root, split, **kwargs):
        self.celeba = CelebA(root, split=split, **kwargs)
        self.attr_names = self.celeba.attr_names[:40]
    
    def __iter__(self):
        for idx in range(len(self.celeba)):
            sample, target = self.celeba[idx]
            labels_by_target = torch.nonzero(target)[:, 0]
            target = ','.join([str(self.attr_names[x]) for x in labels_by_target])
            yield sample, target
    
    def __len__(self):
        return len(self.celeba)

class Flickr8k(IterableDataset):
    def __init__(self, root, transform=None, **kwargs):
        self.data_path = os.path.join(root, "flickr/Flicker8k_Dataset")
        annotations_file = os.path.join(root, "flickr/captions.txt")
        self.flickr8k = []
        self.transform = transform
        
        if not os.path.exists(self.data_path):
            if os.path.exists(os.path.join(root, "flickr/Flicker8k_Dataset.zip")):
                import zipfile
                logger.info("Extracting Flickr8k dataset from zip file...")
                with zipfile.ZipFile(os.path.join(root, "Flicker8k_Dataset.zip"), 'r') as zip_ref:
                    zip_ref.extractall(root)
                logger.info("Flickr8k dataset extracted.")
                #shutil.rmtree(os.path.join(root, "Flicker8k_Dataset.zip"))
                #logger.info("Removed zip file.")
            else:
                logger.error(f"Please download the Flickr8k dataset from https://www.kaggle.com/datasets/adityajn105/flickr8k and place it in {os.path.join(root, 'flickr/Flicker8k_Dataset.zip')}")
                raise RuntimeError("Dataset not found.")
        
        self.split = "train"
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                first_comma = line.index(",")
                image_name = line[:first_comma]
                caption = line[first_comma + 1:]
                self.flickr8k.append((image_name, caption))
    
    def __len__(self):
        return len(self.flickr8k)
    
    def __iter__(self):
        for idx in range(len(self.flickr8k)):
            sample, target = self.flickr8k[idx]
            sample = Image.open(os.path.join(self.data_path, sample))
            sample = sample.convert("RGB")
            if self.transform:
                sample = self.transform(sample)
            yield sample, target


class HFDataset(IterableDataset):
    def __init__(self, dataset, data_path, preprocess, split, download_full=False, **kwargs):
        stream = not download_full
        self.dataset = load_dataset(dataset, cache_dir=data_path, split=split, streaming=stream, trust_remote_code=True, **kwargs)
        self.preprocess = preprocess
        self.len: int = 0
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        for item in self.dataset:
            sample, target = item['jpg'], item['txt']
            if self.preprocess:
                sample = self.preprocess(sample)
            yield sample, target


class ImageNetDataset(HFDataset):
    def __init__(self, data_path, preprocess, split):
        super().__init__("ILSVRC/imagenet-1k", data_path, preprocess, split, False)
        self.len = self.dataset.info.splits[self.dataset.split].num_examples
        self.class_to_idx = {}
        for idx, class_name in enumerate(self.dataset.info.features['label'].names):
            self.class_to_idx[class_name] = idx
        
    def __iter__(self):
        for item in self.dataset:
            sample, target = item['image'], item['label']
            if target == -1:
                target = ""
            if self.preprocess:
                sample = self.preprocess(sample)
            yield sample, target


class CC3MDataset(HFDataset):
    def __init__(self, data_path, preprocess, split, download_full=False):
        super().__init__("pixparse/cc3m-wds", data_path, preprocess, split, download_full)
        if split == "train":
            self.len = 2905954
        else:
            self.len = 13443
    
    def __iter__(self):
        for item in self.dataset:
            sample, target = item['jpg'], item['txt']
            if self.preprocess:
                sample = self.preprocess(sample)
            yield sample, target

class CUBDataset(HFDataset):
    def __init__(self, data_path, preprocess, split, download_full=False):
        super().__init__("cassiekang/cub200_dataset", data_path, preprocess, split, download_full)
        self.len = self.dataset.info.splits[self.dataset.split].num_examples
    
    def __iter__(self):
        for item in self.dataset:
            sample, target = item['image'], item['text']
            if self.preprocess:
                sample = self.preprocess(sample)
            yield sample, target


class CUBA(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose | None = None,
        split: str = "train",
        return_attributes: str = "image",  # 'image' or 'class'
        confidence_level: int = 2,
    ):
        """
        CUB-200-2011 Dataset focusing on attribute labels.
        
        Args:
            root_dir (str): Root directory of the CUB dataset
            transform (transforms.Compose, optional): Image transformations
            split (str): 'train' or 'test'
            return_attributes (str): Whether to return image-level or class-level attributes
        """
        self.root_dir = root_dir
        self.split = split
        self.return_attributes = return_attributes
        self.confidence_level = confidence_level
        if self.return_attributes == "class":
            self.confidence_level *= 25
        
        # Set default transforms if none provided
        self.transform = transform

        # Check if dataset is already downloaded
        if not os.path.exists(os.path.join(self.root_dir, 'CUB_200_2011')):
            if os.path.exists(os.path.join(self.root_dir, 'CUB_200_2011.tgz')):
                import tarfile
                logger.info("Extracting CUB dataset from tar file...")
                tar = tarfile.open(os.path.join(self.root_dir, 'CUB_200_2011.tgz'), "r:gz")
                tar.extractall(self.root_dir)
                tar.close()
                shutil.move(os.path.join(self.root_dir, 'attributes.txt'), os.path.join(self.root_dir, 'CUB_200_2011'))
                logger.info("CUB dataset extracted.")
                #shutil.rmtree(os.path.join(self.root_dir, 'CUB_200_2011.tgz'))
                #logger.info("Removed tar file.")
            else:
                logger.error(f"Please download the CUB dataset from https://www.vision.caltech.edu/datasets/cub_200_2011/ and place it in {os.path.join(self.root_dir, 'CUB_200_2011.tgz')}")
                raise RuntimeError("Dataset not found.")
        
        self.root_dir = os.path.join(self.root_dir, 'CUB_200_2011')

        # Load metadata
        self.images_df = self._load_images_data()
        self.class_labels = self._load_class_labels()
        self.attributes = self._load_attributes()
        self.certainty = self._load_certainty()
        self.classes = self._load_classes()
        self.image_attributes = None
        self.class_attributes = None
        if self.return_attributes == "image":
            self.image_attributes = self._load_image_attributes()
        else:
            self.class_attributes = self._load_class_attributes()
        
        
        # Filter by split
        self.split_info = self._load_split_info()
        split_mask = self.split_info['is_training'] == (1 if split == 'train' else 0)
        self.images_df = self.images_df[split_mask].reset_index(drop=True)
    
    def _load_images_data(self) -> pd.DataFrame:
        """Load image paths and IDs."""
        images_file = os.path.join(self.root_dir, 'images.txt')
        return pd.read_csv(images_file, sep=' ', names=['image_id', 'filepath'])
    
    def _load_class_labels(self) -> pd.DataFrame:
        """Load image-to-class mappings."""
        labels_file = os.path.join(self.root_dir, 'image_class_labels.txt')
        return pd.read_csv(labels_file, sep=' ', names=['image_id', 'class_id'])
    
    def _load_attributes(self) -> pd.DataFrame:
        """Load attribute names and IDs."""
        attributes_file = os.path.join(self.root_dir, 'attributes.txt')
        df = pd.read_csv(attributes_file, sep=' ', names=['attribute_id', 'attribute_name'])
        df['attribute_name'] = df['attribute_name'].apply(lambda x: x.replace("has_", " ").replace("_", " ").replace("::", " "))
        return df

    def _load_certainty(self) -> pd.DataFrame:
        """Load certainty levels."""
        certainty_file = os.path.join(self.root_dir, 'attributes', 'certainties.txt')
        with open(certainty_file, 'r') as f:
            lines = f.readlines()
            certainty = []
            for line in lines:
                line = line.strip()
                first_space = line.index(" ")
                certainty.append(line[first_space + 1:])
    
        return certainty
    
    def _load_classes(self) -> pd.DataFrame:
        """Load class names."""
        classes_file = os.path.join(self.root_dir, 'classes.txt')
        df = pd.read_csv(classes_file, sep=' ', names=['class_id', 'class_name'])
        df['class_name'] = df['class_name'].apply(lambda x: x[x.index(".") + 1:])
        return df
    
    def _load_image_attributes(self) -> pd.DataFrame:
        """Load image-level attribute labels."""
        attributes_file = os.path.join(
            self.root_dir, 'attributes', 'image_attribute_labels.txt'
        )

        data = []
        with open(attributes_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split()
                image_id = int(parts[0])
                attribute = self.attributes[self.attributes['attribute_id'] == int(parts[1])]['attribute_name'].values[0].strip()
                is_present = int(parts[2])
                certainty = int(parts[3])
                data.append((image_id, attribute, is_present, certainty))
        
        df = pd.DataFrame(data, columns=['image_id', 'attribute', 'is_present', 'certainty'])
        
        # filter only attributes is present
        df = df[df["is_present"] > 0]
        return df
    
    def _load_class_attributes(self) -> np.ndarray:
        """Load class-level attribute labels."""
        attributes_file = os.path.join(
            self.root_dir, 
            'attributes',
            'class_attribute_labels_continuous.txt'
        )
        data = []
        with open(attributes_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split(" ")
                data.append(parts)
        
        
        # Convert to numpy array and normalize to [0, 1]
        return np.array(data, dtype=float)
    
    def _load_split_info(self) -> pd.DataFrame:
        """Load train/test split information."""
        split_file = os.path.join(self.root_dir, 'train_test_split.txt')
        return pd.read_csv(split_file, sep=' ', names=['image_id', 'is_training'])
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.images_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, attribute_labels, class_id)
                - image_tensor: Transformed image
                - attribute_labels: Either image-level or class-level attributes
                - class_id: Class label for the image
        """
        # Get image data
        row = self.images_df.iloc[idx]
        image_path = os.path.join(self.root_dir, 'images', row['filepath'])
        image_id = row['image_id']
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get class label
        class_id = self.class_labels[
            self.class_labels['image_id'] == image_id
        ]['class_id'].iloc[0]
        
        # Get attribute labels
        if self.return_attributes == 'image':
            attributes = self.image_attributes[(self.image_attributes['certainty'] >= self.confidence_level) & (self.image_attributes['image_id'] == image_id)]['attribute'].values.tolist()
        else:
            attributes = []
            for idx, value in enumerate(self.class_attributes[class_id-1]):
                if value >= self.confidence_level:
                    attributes.append(self.attributes['attribute_id'].iloc[idx])
    
        attributes.append(self.classes[self.classes['class_id'] == class_id]['class_name'].values[0])
        attributes = ', '.join(attributes)
        
        return image, attributes


def load(dataset, preprocess, data_path, train=False, download=False):
    os.makedirs(data_path, exist_ok=True)
    
    if dataset == "CIFAR10":
        dataset_test = CIFAR10(data_path, download=download, train=train, transform=preprocess)

    elif dataset == "CIFAR100":
        dataset_test = CIFAR100(data_path, download=download, train=train, transform=preprocess)

    elif dataset == "CelebA":
        dataset_test = CelebAMy(data_path, download=download, split="train" if train else "test", transform=preprocess, target_type="attr")
    
    elif dataset == "Places365":
        dataset_test = Places365(data_path, download=download, transform=preprocess, split="train-standard" if train else "val", small=True)
        dataset_test.class_to_idx = {k[3:]: v for k, v in dataset_test.class_to_idx.items()}
        
    elif dataset == "ImageNet":
        dataset_test = ImageNetDataset(data_path, preprocess, "train" if train else "validation")
    
    elif dataset == "CC3M":
        dataset_test = CC3MDataset(data_path, preprocess, "train" if train else "validation")
    
    elif dataset == "Flickr8k":
        dataset_test = Flickr8k(data_path, download=download, transform=preprocess)
    
    elif dataset == "CUB":
        dataset_test = CUBDataset(data_path, preprocess, "train" if train else "test")
    
    elif dataset == "CUBA":
        dataset_test = CUBA(data_path, transform=preprocess, split="train" if train else "test", return_attributes="image", confidence_level=3)
    
    else:
        raise RuntimeError(f"Dataset {dataset} not supported.")

    if hasattr(dataset_test, "class_to_idx"):
        dataset_test.idx_to_class = {v: k for k, v in dataset_test.class_to_idx.items()}

    return  dataset_test

def load_vocab(vocab, vocab_size = -1):
    if vocab == "mscoco":
        path = "data/vocab/mscoco_unigram.txt"
    elif vocab == "laion_unigram":
        path = "data/vocab/laion_400_unigram.txt"
    elif vocab == "laion_bigrams":
        path = "data/vocab/laion_400_bigram.txt"
    elif vocab == "laion":
        unigram_data = load_vocab("laion_unigram", vocab_size // 2 if vocab_size > 0 else -1)
        bigram_data = load_vocab("laion_bigrams", vocab_size // 2 if vocab_size > 0 else -1)
        return unigram_data + bigram_data
    else:
        raise RuntimeError(f"Vocab {vocab} not supported.")
    
    vocab_data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        if vocab_size > 0:
            if vocab_size > len(lines):
                warnings.warn(f"Vocab size {vocab_size} is greater than the actual vocab size {len(lines)}. Using full vocab.")
            else:
                lines = lines[-vocab_size:]

        for line in lines:
            line = line.strip()
            vocab_data.append(line)
    
    print(f"Loaded {len(vocab_data)} vocab items from {path}")
    return VocabDataset(vocab_data)