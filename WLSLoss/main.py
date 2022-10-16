import torch
from PIL import Image
import numpy as np
from wlsloss import Standard_WLSLoss
from torchvision.transforms import ToTensor

img = Image.open('1.png')

img2tensor = ToTensor()

img = img2tensor(img)

loss = Standard_WLSLoss()

ax,ay = loss(img)

ax = np.around(ax.numpy(), 2)
ay = np.around(ay.numpy(), 2)

print(ax[0])

np.savetxt('./data/ax.csv', ax[0])
np.savetxt('./data/ay.csv', ay[0])
