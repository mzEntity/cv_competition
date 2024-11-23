import torch
import time
import os
import shutil
from model.model import MyModel
from common.datasets import get_dataset

def train(device, train_loader, validate_loader, model, optimizer, criterion, epoch, save_root):
    for epoch in range(1, epoch+1):
        print("========== new epoch ==========")
        # train one epoch
        epoch_start_time = time.time()        
        model.train()
        
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - epoch_start_time
        print('Epoch time: {}s'.format(int(epoch_duration)))

        print('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'net': model.state_dict()
        }, save_root, epoch)

        print("validating...")
        validate(device, validate_loader, model, criterion)


def validate(device, loader, model, criterion):
    model.eval()
    
    loss_list = []
    for i, (img, label) in enumerate(loader):
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
            
            loss_list.append(loss)

    print(loss_list)


def save_checkpoint(state, save_root, epoch):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, 'epoch_{}.pth.tar'.format(str(epoch)))
    torch.save(state, save_path)

if __name__ == "__main__":
    root_dir = 'D://fudan//2024Autumn//CV//competition//cv_competition'
    batch_size = 32
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel(classes_num=100).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters())
    
    train_dataset = get_dataset(root_dir, "train")
    validate_dataset = get_dataset(root_dir, "validate")
    test_dataset = get_dataset(root_dir, "test")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    
    train(device, train_loader, validate_loader, model, optimizer, criterion, 10, os.path.join(root_dir, "save"))