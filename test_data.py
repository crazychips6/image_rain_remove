import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class MyTestDataset(Dataset):
    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path
        self.input_files = os.listdir(self.input_path)
        self.transforms = transforms.Compose([
            # transforms.CenterCrop([128,128]),#这行没有必要
            transforms.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_image_path = os.path.join(self.input_path, self.input_files[index])
        input_image = Image.open(input_image_path).convert('RGB')
        inputs = self.transforms(input_image)
        return inputs
