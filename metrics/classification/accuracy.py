import torch


class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')


import numpy as np 

class AverageAccuracy():
    def __init__(self, nclasses, **kwargs):
        self.nclasses = nclasses
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)

        pred[pred != target] = 0
        
        freq = torch.bincount(pred, minlength=9)
        freq_tar = torch.bincount(target, minlength=9)
        return (freq, freq_tar) #correct, sample_size

    def update(self, value):
        freq, freq_tar = value 
        self.correct += freq.cpu().numpy()[:8]
        self.sample_size += freq_tar.cpu().numpy()[:8]

    def reset(self):
        self.correct = np.zeros(self.nclasses + 1)
        self.sample_size = np.zeros(self.nclasses + 1)


    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')
