import torch
import torchvision
import torch.nn.functional as F
# from dropblock import drop_block_2d

# Define a simple convolutional neural network
class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torchvision.ops.drop_block2d(x, p=0.2, block_size=5)
        # x = drop_block_2d(x, drop_prob=0.2, block_size=5)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torchvision.ops.drop_block2d(x, p=0.2, block_size=5)
        # x = drop_block_2d(x, drop_prob=0.2, block_size=5)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x

# Create a random input tensor
x = torch.randn(128, 3, 32, 32)

# Create an instance of the CNN and apply it to the input
model = MyCNN()
y = model(x)

# Compute the loss and backpropagate
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(y, torch.randint(10, (128,)))
loss.backward()