import torch
from tqdm import trange
from torch import nn

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='../Checkpoint/selected.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, train_loader, valid_loader, lr, epochs, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    valid_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (trange(epochs) as tr):
        for epoch in tr:
            # Training phase
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                # for p in model.decoder_out.parameters():
                #     p.data.clamp_(0)
                # for p in model.encoder_out.parameters():
                #     p.data.clamp_(0)
                # for p in model.encoder_hvac.parameters():
                #     p.data.clamp_(0)
                # for p in model.decoder_hvac.parameters():
                #     p.data.clamp_(0)

                train_loss += loss.item() * data.size(0)

            # Validation phase
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    valid_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
                           valid_loss="{0:.6f}".format(valid_loss))

            # Early Stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('../Checkpoint/selected.pt'))
        return model, train_losses, valid_losses

def test_model(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
    return outputs.cpu()