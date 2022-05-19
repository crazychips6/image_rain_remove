import torch
import os
from prn import Net
from test_data import MyTestDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试图像的路径
input_path = "./test_data"

if not os.path.exists('output'):
    os.makedirs('output')

num_files = len(os.listdir(input_path))
net = Net().to(device)
net.load_state_dict(torch.load('./model.pth'))
dataloader = DataLoader(MyTestDataset(input_path))

for i, inputs in enumerate(dataloader):
    net.eval()
    inputs = inputs.to(device)
    print('finished:{:.2f}%'.format((i + 1) * 100 / num_files))
    with torch.no_grad():
        output_image = net(inputs)
        save_image(output_image, './output/' + str(i + 1).zfill(3) + '.jpg')
