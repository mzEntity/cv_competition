import torch
import time
import os
import shutil
from model.model import MyModel, MyResNetModel, MyResNetModel2
from common.datasets import get_dataset
import torch.optim as optim

from tqdm import tqdm

def train(device, train_loader, validate_loader, model, optimizer, criterion, max_epochs, save_path):
    """
    Train the model for regression with early stopping based on validation loss.

    Args:
        device (torch.device): The device to use for training (e.g., "cuda" or "cpu").
        train_loader (DataLoader): DataLoader for the training dataset.
        validate_loader (DataLoader): DataLoader for the validation dataset.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion (nn.Module): Loss function for regression (e.g., MSELoss).
        max_epochs (int): Maximum number of epochs to train.
        save_path (str): Directory to save the best model.
    """
    best_loss = float('inf')
    patience = 3  # Stop if no improvement for 3 consecutive epochs
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for (names, images), labels in tqdm(train_loader, desc="Epoch_" + str(epoch) + " Train Processing:"):
            images, labels = images.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(images).squeeze(1)  # Ensure outputs are 1D
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        validate_loss = 0.0

        with torch.no_grad():
            for (names, images), labels in validate_loader:
                images, labels = images.to(device), labels.to(device).float()

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                validate_loss += loss.item()

        avg_validate_loss = validate_loss / len(validate_loader)
        print(f"Validation Loss: {avg_validate_loss:.4f}")

        is_best = False
        # Check for improvement
        if avg_validate_loss < best_loss:
            print("Validation loss improved, saving model...")
            is_best = True
            best_loss = avg_validate_loss
            patience_counter = 0
        else:
            patience_counter += 1
        save_checkpoint(model.state_dict(), save_path, is_best, epoch)
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered. Best validation loss:", best_loss)
            break

    print("Training complete. Best validation loss:", best_loss)



def save_checkpoint(state, save_root, is_best, epoch):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, 'epoch_{}.pth.tar'.format(str(epoch)))
    torch.save(state, save_path)
    
    best_path = os.path.join(save_root, 'best_model.pth.tar'.format(str(epoch)))
    torch.save(state, best_path)

if __name__ == "__main__":
    print("Start...")
    root_dir = 'D://fudan//2024Autumn//CV//competition//cv_competition'
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyResNetModel2().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = get_dataset(root_dir, "train")
    validate_dataset = get_dataset(root_dir, "validate")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    
    
    train(
        device,
        train_loader,
        validate_loader,
        model,
        optimizer,
        criterion,
        max_epochs=50,
        save_path=os.path.join(root_dir, "save")
    )
