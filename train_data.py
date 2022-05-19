import os
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class MyTrainDataset(Dataset):
    def __init__(self, input_path, label_path):
        self.input_path = input_path
        self.input_files = os.listdir(input_path)
        self.label_path = label_path
        self.label_files = os.listdir(label_path)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop([64, 64]),
            torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_image_path = os.path.join(self.input_path, self.input_files[index])
        input_image = Image.open(input_image_path).convert('RGB')

        label_image_path = os.path.join(self.label_path, self.label_files[index])
        label_image = Image.open(label_image_path).convert('RGB')

        inputs = self.transforms(input_image)
        label = self.transforms(label_image)
        return inputs, label
