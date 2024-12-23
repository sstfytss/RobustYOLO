## l0_attack.py -- attack a network optimizing for l_0 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt


MAX_ITERATIONS = 10   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
INITIAL_CONST = 1e-3    # the first value of c to start at
LARGEST_CONST = 20     # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = False        # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CarliniL0():
    def __init__(self, model, num_labels = 10, image_size = 32, num_channels = 1, batch_size = 1,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 reduce_const = REDUCE_CONST, const_factor = CONST_FACTOR,
                 independent_channels = False):
        """
        The L_0 optimized attack. 
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        independent_channels: set to false optimizes for number of pixels changed,
          set to true (not recommended) returns number of channels changed.
        """

        self.model = model
        self.num_labels = num_labels
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const    # c
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

       
    def doit(self, oimgs, labs, starts, valid, CONST):
        # convert to tanh-space
        
        imgs = torch.atanh(oimgs)
        starts = torch.atanh(starts)

        shape = (self.batch_size, self.num_channels, self.image_size, self.image_size)
        
        # the variable to optimize over
        modifier = torch.nn.parameter.Parameter(torch.zeros(shape, dtype=torch.float32, device=device), requires_grad=True)

        # the variables we're going to hold, use for efficiency
        canchange = valid.to(device)
        simg = starts.to(device)
        original = oimgs.to(device)
        timg = imgs.to(device)
        tlab = labs.to(device)
        const = []

        tlab = torch.nn.functional.one_hot(tlab, num_classes= 29)
        
        optimizer = torch.optim.Adam([modifier], lr=self.LEARNING_RATE)

        while CONST < self.LARGEST_CONST:
            # try solving for each value of the constant
            # print('try const', CONST)
            for step in range(self.MAX_ITERATIONS):

                # remember the old value
                oldmodifier = modifier
                
                newimg = (torch.tanh(modifier + simg)/2)*canchange+(1-canchange)*original
                
                output = self.model(newimg)
                
                real = torch.sum((tlab)*output,-1)
                other = torch.maximum(torch.max((1-tlab)*output - (tlab*10000)), torch.tensor(1.0, device=device))

                if self.TARGETED:
                    # if targetted, optimize for making the other class most likely
                    loss1 = torch.maximum(torch.tensor(0.0, device=device), other-real+.01)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = torch.maximum(torch.tensor(0.0, device=device), real-other+.01)

                # sum up the losses
                loss2 = torch.sum(torch.square(newimg-torch.tanh(timg)/2))
                loss = CONST*loss1+loss2
                
                works = loss1
                scores = output
                
                # if step%(self.MAX_ITERATIONS//1) == 0:
                #     print(step,loss1,loss2)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if torch.all(scores>=-.0001) and torch.all(scores <= 1.0001):
                    if torch.allclose(torch.sum(scores,axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                    
                if works < .0001 and self.ABORT_EARLY:
                    # it worked previously, restore the old value and finish
                    
                    modifier = oldmodifier
                    grads = torch.autograd.grad(loss, modifier, torch.ones_like(loss))[0]
                    nimg = (torch.tanh(modifier + simg)/2)*canchange+(1-canchange)*original

                    l2s=torch.square(nimg-torch.tanh(imgs)/2).sum(axis=(1,2,3))
                    return grads, scores, nimg, CONST

            # we didn't succeed, increase constant and try again
            CONST *= self.const_factor
        
    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = torch.tensor([], device=device)
        for i,(img,target) in enumerate(zip(imgs, targets)):
            # print("Attack iteration",i)
            r = torch.cat((r, self.attack_single(img, target)), dim=0)
        return r

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        valid = torch.ones((self.batch_size,self.num_channels,self.image_size,self.image_size))

        # the previous image
        prev = img.clone().reshape((self.batch_size,self.num_channels,self.image_size,self.image_size))

        # initially set the solution to None, if we can't find an adversarial
        # example then we will return None as the solution.
        last_solution = img.unsqueeze(0)
        const = self.INITIAL_CONST

        equal_count = None
    
        while True:
            # try to solve given this valid map
            res = self.doit(img.clone(), target, prev.clone(), 
                       valid, const)
            if res == None:
                # the attack failed, we return this as our final answer
                # print("Final answer",equal_count)
                return last_solution
    
            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg, const = res
            if self.REDUCE_CONST: const /= 2
            
            equal_count = self.num_channels*self.image_size**2-torch.sum(torch.all(torch.abs(img-nimg[0])<.0001, dim=0, keepdim=True))
            # print("Forced equal:",torch.sum(1-valid),
            #       "Equal count:",equal_count)
            if torch.sum(valid) == 0:
                # if no pixels changed, return 
                return img.unsqueeze(0)
    
            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = torch.abs(nimg[0]-img)*torch.abs(gradientnorm[0])
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.flatten()
                totalchange = torch.abs(torch.sum(nimg[0]-img,axis=0))*torch.sum(torch.abs(gradientnorm[0]),axis=0)
            totalchange = totalchange.flatten()

            # set some of the pixels to 0 depending on their total change
            did = 0
            for e in torch.argsort(totalchange):
                if torch.all(valid[e]):
                    did += 1
                    valid[e] = 0

                    if totalchange[e] > .01:
                        # if this pixel changed a lot, skip
                        break
                    if did >= .3*equal_count**.5:
                        # if we changed too many pixels, skip
                        break

            valid = torch.reshape(valid,(self.batch_size,self.num_channels, self.image_size,self.image_size))
            # print("Now forced equal:",torch.sum(1-valid))
    
            last_solution = prev = nimg


# Assuming CarliniL0 is already defined (the code you shared above)

# Step 1: Load or define your model (an example model for illustration)
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.fc = nn.Linear(10 * 416 * 416, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

script_dir = Path(__file__).resolve().parent
model_path = script_dir.parent / 'Models' / 'RobustYOLO/Models/YOLOv8_Trained_50.pt'
# Load your pre-trained model here
model = YOLO('../Models/YOLOv8_Trained_50.pt')

# Step 2: Initialize the CarliniL0 attack
attack = CarliniL0(model=model, num_labels=29, image_size=416, num_channels=3, batch_size=1)

# Define paths
image_dir = "../Dataset/images/test"
label_dir = "../Dataset/labels/test"

# Step 1: Get a list of image files (assuming all are .jpg for illustration, you can adjust as needed)
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Step 2: Pick a specific image file (for demonstration, we will choose the first one)
selected_image_file = image_files[0]

# Step 3: Load the image using PIL
image_path = os.path.join(image_dir, selected_image_file)
image = Image.open(image_path)

# Step 4: Display the image
# plt.imshow(image)
# plt.title("Loaded Test Image")
# plt.axis('off')
# plt.show()

# Step 5: Load the corresponding label
# Assuming the label file has the same name but with a '.txt' extension, following YOLO format
label_file = selected_image_file.replace(".jpg", ".txt")
label_path = os.path.join(label_dir, label_file)

# Step 6: Read the label file
with open(label_path, 'r') as f:
    label_data = f.readlines()

# Step 7: Display the label data
print(f"Label data for {selected_image_file}:")
for line in label_data:
    print(line.strip())


# Step 3: Prepare the input image
image_np = np.array(image).astype(np.float32) / 255.0
image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)

parsed_label = label_data[0].strip().split()  # Take the first line for illustration
target_label = torch.tensor([int(parsed_label[0])])

# Step 5: Generate the adversarial image
adversarial_tensor = attack.attack(image_tensor, target_label)

# # Step 6: Convert tensors back to numpy arrays for visualization
# original_image_np = image_tensor.squeeze().permute(1, 2, 0).detach().numpy()
# adversarial_image_np = adversarial_tensor.squeeze().permute(1, 2, 0).detach().numpy()

# # Clip the adversarial image values to be between 0 and 1 (to visualize properly)
# adversarial_image_np = np.clip(adversarial_image_np, 0, 1)

# # Step 7: Display the images using Matplotlib
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(original_image_np)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Adversarial Image")
# plt.imshow(adversarial_image_np)
# plt.axis('off')

# plt.show()

