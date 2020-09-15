from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, ToTensor
from PIL import Image
import numpy as np
import os

rootDir = "/home/adi/practice/cs231n/projects/texture/dtd"
imgFolder = "images"
jointLabels = "labels/labels_joint_anno.txt"

# can experiment with different transforms - random crops, resizing, ton of hyperparameters in the transform, not even including the model itself
# random crop, resize, center crop, padding
# number of labels to use
class TexturesDataset(Dataset):

    def __init__(self, rootDir, imgFolder, jointLabels, transform = None):

        self.rootDir = rootDir
        self.imgFolder = imgFolder
        self.jointLabels = jointLabels
        self.transform = transform
        self.jpgs = []
        self.label1 = []
        self.label2 = []
        self.label3 = []

        try:
            with open(os.path.join(self.rootDir, self.jointLabels), 'r') as f:
                files = f.read().splitlines()
        except e:
                print("Unable to read file")
                exit()


        for f in files:
            try:
                jpgLabel = f.split()
                self.jpgs.append(jpgLabel[0])
                label = jpgLabel[1:]
                # max number of labels is 4. make sure to document the reasons why you do things in some type of side document
                while len(label) < 3:
                    label.append("")

                self.label1.append(label[0])
                self.label2.append(label[1])
                self.label3.append(label[2])
            except e:
                print("Unable to parse")


        #             print(self.jpgs[23])
                    # textures = set([])
                    # for tex in self.Labels:
                    #     for val in tex:
                    #         textures.add(val)
                    #
                    #
                    #         idxDict = {}
                    #         for idx, tex in enumerate(textures):
                    #             idxDict[tex] = idx
                    #             return idxDict
                    #
                    #             self.idxDict = self.labelToIdx(textures)
                    #
                    #             self.numLabels = len(textures)

    def __len__(self):
        return len(self.jpgs)

    def __getitem__(self, idx):

        PATH = os.path.join(self.rootDir, self.imgFolder, self.jpgs[idx])

        img = Image.open(PATH)

        # label = np.zeros(len(self.numLabels))
        #
        # for val in self.Labels[idx]:
        #     label[self.idxDict[val]] = 1

        if self.transform:
            self.transform(img)

        data = {"img": img, "labels": {"label1": self.label1[idx], "label2": self.label2[idx], "label3": self.label3[idx]}}

        return data

b = TexturesDataset(rootDir, imgFolder, jointLabels, transform = None)
