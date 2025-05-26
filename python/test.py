import torch
from torch.utils.data import DataLoader, Dataset
from model import SiameseLSTM
from train import load_data, evaluate, SiameseDataset

def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data
    pairs, labels = load_data('X_Y_val20_pairs.pkl')

    # Create test dataset and data loader
    test_dataset = SiameseDataset(pairs, labels)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Set up model
    input_dim = pairs.shape[-1]
    hidden_dim = 128  # Adjust if needed
    dropout_rate = 0.2 
    model = SiameseLSTM(input_dim, hidden_dim, dropout=dropout_rate).to(device)
    
    # Load pre-trained model weights
    pretrained_dict = torch.load('siamese_model.pth')
    model.load_state_dict(pretrained_dict)  # Load the state dictionary directly

    # Evaluate the model on the test set
    criterion = torch.nn.BCEWithLogitsLoss()  # Use the same criterion as in training
    test_loss, test_acc = evaluate(model, criterion, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    test_model()