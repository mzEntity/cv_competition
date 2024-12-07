import torch
from model.model import MyModel, MyResNetModel, MyResNetModel2
from common.datasets import get_dataset
import os
from tqdm import tqdm

def test(device, test_loader, model, criterion):
    """
    Evaluate the model on the test set.

    Args:
        device (torch.device): The device to use for testing (e.g., "cuda" or "cpu").
        test_loader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): The trained model.
        criterion (nn.Module): Loss function for regression (e.g., MSELoss).

    Returns:
        float: The average test loss.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    total_samples = 0
    total_correct = 0  # For accuracy or other metrics if needed

    with torch.no_grad():  # No need to track gradients during testing
        with open('output.txt', 'w', encoding='utf-8') as file:
            for (names, images), labels in tqdm(test_loader, desc="Eval Processing:"):
                images, labels = images.to(device), labels.to(device).float()

                # Forward pass
                outputs = model(images).squeeze(1)  # Ensure outputs are 1D
                
                for i, name in enumerate(names):
                    file.write(f"{name}\t{round(outputs[i].item())}\n")
                
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                # If you want to compute additional metrics, like MSE or MAE, you can do so here
                # For example, to compute MAE:
                total_samples += len(labels)
                total_correct += torch.abs(outputs - labels).sum().item()  # Absolute error

    avg_test_loss = test_loss / len(test_loader)
    avg_mae = total_correct / total_samples  # Mean Absolute Error

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test MAE: {avg_mae:.4f}")

    return avg_test_loss, avg_mae


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = 'D://fudan//2024Autumn//CV//competition//cv_competition'
    batch_size = 32
    model = MyResNetModel2()

    # 2. 加载模型权重
    model_path = os.path.join(root_dir, "save/best_model.pth.tar")  # 模型保存路径
    model.load_state_dict(torch.load(model_path))  # 加载权重到模型
    model.to(device)
    
    test_dataset = get_dataset(root_dir, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)

    criterion = torch.nn.MSELoss()
    
    # 调用测试函数
    avg_test_loss, avg_mae = test(device, test_loader, model, criterion)
    print(avg_test_loss, avg_mae)    
