
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import os
import torch
import os.path
from torchvision.utils import save_image
    
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)

    for target in sorted(os.listdir(dir)):

        d = os.path.join(dir, target)
        
        if not os.path.isdir(d):
            continue
    
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):

                    path = os.path.join(root, fname)
                    path_sec = path.replace('/adv/', '/clean/')
                    # path_thr = path.replace('/adv/', '/gen0/')

                    item = (path, path_sec, class_to_idx[target])
                    
                    images.append(item)
    
    return images

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class MyCustomDataset(Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        (root/train/clean/class_x/xxx.ext, root/train/adv/class_x/xxx.ext, class_x)

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path_adv, path_clean, target = self.samples[index]
        
        sample_clean = self.loader(path_clean)
        sample_adv = self.loader(path_adv)
        # sample_gen0 = self.loader(path_gen0)
        if self.transform is not None:
            sample_clean = self.transform(sample_clean)
            sample_adv = self.transform(sample_adv)
            # sample_gen0 = self.transform(sample_gen0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample_clean, sample_adv, target

    def __len__(self):
        return len(self.samples)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFoldery(MyCustomDataset):
  
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
                 
        super(ImageFoldery, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

   



#----------------------
# use for step training
#----------------------
def set_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    
    for target in sorted(os.listdir(dir)):

        d = os.path.join(dir, target)
        
        if not os.path.isdir(d):
            continue
        
        if target == 'adv':
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        path_sec = path.replace('/adv/', '/clean/')
                        item = (path, path_sec, class_to_idx[target])
                        
                        images.append(item)
                  
    return images

class CustomGenerateDataset(Dataset):

    """
    used for generate step training
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        
        classes, class_to_idx = find_classes(root)
        
        samples = set_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path_clean, path_adv, target = self.samples[index]
        sample_clean = self.loader(path_clean)
        sample_adv = self.loader(path_adv)
        if self.transform is not None:
            sample_clean = self.transform(sample_clean)
            sample_adv = self.transform(sample_adv)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample_clean, sample_adv, path_clean, target

    def __len__(self):
        return len(self.samples)



class GenerateFoldery(CustomGenerateDataset):
  
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(GenerateFoldery, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples