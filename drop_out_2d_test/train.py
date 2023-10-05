import torch
import torchvision
import torchvision.transforms as transforms
from ResNet import ResNet18, ResNet18Drop

# Load CIFAR-10 dataset
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root="/home/lms/code/learnCode/data/",
    train=True,
    download=False,
    transform=transform_train,
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root="/home/lms/code/learnCode/data/",
    train=False,
    download=False,
    transform=transform_test,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

# Create an instance of the ResNet18 model
net = ResNet18().to("cuda")
net.load_state_dict(torch.load("/home/lms/code/learnCode/drop_out_2d_test/best_model.pth")) # 导入网络的参数
net2 = ResNet18Drop().to("cuda")
net2.load_state_dict(torch.load("/home/lms/code/learnCode/drop_out_2d_test/best_model_with_drop.pth")) # 导入网络的参数

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Train the model
for epoch in range(100):
    running_loss = 0.0
    running_loss2 = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()

        optimizer2.zero_grad()

        # Forward pass
        outputs = net(inputs)
        outputs2 = net2(inputs)

        loss = criterion(outputs, labels)
        loss2 = criterion(outputs2, labels)

        # Backward pass and update weights
        loss.backward()
        loss2.backward()

        optimizer.step()
        optimizer2.step()

        # Print statistics
        running_loss += loss.item()
        running_loss2 += loss2.item()

        if i % 100 == 99:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            print(
                "[%d, %5d] loss with drop: %.3f"
                % (epoch + 1, i + 1, running_loss2 / 100)
            )
            running_loss, running_loss2 = 0.0, 0.0
    # print accuracy on training set
    correct = 0
    correct2 = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs, outputs2 = net(images), net2(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()
    print("Accuracy on training set: %d %%" % (100 * correct / total))
    print("Accuracy with drop on training set: %d %%" % (100 * correct2 / total))
    # print accuracy on test set
    correct = 0
    correct2 = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs, outputs2 = net(images), net2(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))
    print("Accuracy with drop on test set: %d %%" % (100 * correct2 / total))

    # save the best model
    if epoch == 0:
        best_acc = 100 * correct / total
        torch.save(net.state_dict(), "best_model.pth")
        best_acc2 = 100 * correct2 / total
        torch.save(net2.state_dict(), "best_model_with_drop.pth")
    else:
        if 100 * correct / total > best_acc:
            best_acc = 100 * correct / total
            torch.save(net.state_dict(), "best_model.pth")
            best_acc2 = 100 * correct2 / total
            torch.save(net2.state_dict(), "best_model_with_drop.pth")
