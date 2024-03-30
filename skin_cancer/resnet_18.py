from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    # Create transform function
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dir = "./skin_cancer/data/train/"
    test_dir = "./skin_cancer/data/test/"
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features 
    print(num_features)

    model.fc = torch.nn.Linear(512, 2)
    model = model.to('mps')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


    train_loss=[]
    train_accuary=[]
    test_loss=[]
    test_accuary=[]

    num_epochs = 10
    start_time = time.time() 

    for epoch in range(num_epochs): 
        print("Epoch {} running".format(epoch)) 
        """ Training Phase """
        model.train()  
        running_loss = 0.
        running_corrects = 0 

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to('mps')
            labels = labels.to('mps') 

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.

        train_loss.append(epoch_loss)
        train_accuary.append(epoch_acc)
        # Print progress
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))

        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to('mps')
                labels = labels.to('mps')
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset) * 100.

            test_loss.append(epoch_loss)
            test_accuary.append(epoch_acc)

            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))

    pyplot.figure(figsize=(6,6))
    pyplot.plot(numpy.arange(1,num_epochs+1), train_accuary,'-o')
    pyplot.plot(numpy.arange(1,num_epochs+1), test_accuary,'-o')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend(['Train','Test'])
    pyplot.title('Train vs Test Accuracy over time')
    pyplot.show()

    pyplot.figure(figsize=(6,6))
    pyplot.plot(numpy.arange(1,num_epochs+1), train_loss,'-o')
    pyplot.plot(numpy.arange(1,num_epochs+1), test_loss,'-o')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend(['Train','Test'])
    pyplot.title('Train vs Test Accuracy over time')
    pyplot.show()