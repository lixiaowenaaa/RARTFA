# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""

import random

from torchvision import transforms
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, show_step=10, image_size=30):
        self.transform = transforms.Compose([
            # transforms.Normalize(mean = [-2.118, -2.036, -1.804], # Equivalent to un-normalizing ImageNet (for correct visualization)
            #                                                         std = [4.367, 4.464, 4.444]),
                                            transforms.Normalize((0,), (1,)),
                                            transforms.ToPILImage(),
                                            transforms.Scale(image_size)])

        self.show_step = show_step
        self.step = 0
        
        # self.figure, (self.lr_plot, self.hr_plot,self.gen0_plot, self.gen1_plot, self.gen2_plot) = plt.subplots(1,5,figsize=(10,5))
        self.figure, (self.lr_plot, self.hr_plot,self.gen0_plot) = plt.subplots(1,3,figsize=(10,5))
    
        self.figure.show()

        self.lr_image_ph = None
        self.hr_image_ph = None
        
        self.gen0 = None
        # self.gen1 = None
        # self.gen2 = None

    def show(self, inputsG, inputsD_real, gen0):

        self.step += 1
        if self.step == self.show_step:
            self.step = 0

            i = random.randint(0, inputsG.size(0) -1)

            lr_image = self.transform(inputsG[i])
            hr_image = self.transform(inputsD_real[i])
            
            fake_gen0 = self.transform(gen0[i])
            # fake_gen1 = self.transform(gen1[i])
            # fake_gen2 = self.transform(gen2[i])

            if self.lr_image_ph is None:
                self.lr_image_ph = self.lr_plot.imshow(lr_image)
                self.hr_image_ph = self.hr_plot.imshow(hr_image)
                self.gen0 = self.gen0_plot.imshow(fake_gen0)
                # self.gen1 = self.gen1_plot.imshow(fake_gen1)
                # self.gen2 = self.gen2_plot.imshow(fake_gen2)
            
            else:
                self.lr_image_ph.set_data(lr_image)
                self.hr_image_ph.set_data(hr_image)
                self.gen0.set_data(fake_gen0)
                # self.gen1.set_data(fake_gen1)
                # self.gen2.set_data(fake_gen2)

            self.figure.canvas.draw()
