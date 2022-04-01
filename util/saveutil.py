import torch
import matplotlib.pyplot as plt
import os

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.best_model_path = ''

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, save_dir, fold
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            save_filename = 'best-fold%d.pth'%fold
            save_path = os.path.join(save_dir, save_filename)
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path)
            print(save_filename, 'saved!')
            self.best_model_path = save_path

    def get_modelpath(self):
        return self.best_model_path

def save_plots(loss_dct, save_dir, fold, runname):
    """
    Function to save the loss and accuracy plots per fold.
    ToDO: needs cleaning!
    """
    colors = ['orange', 'red', 'green', 'blue']

    if len(loss_dct['labeled_loss']) > 0:

        plt.figure(figsize=(10, 7))
        plt.plot(loss_dct['labeled_loss'],
                 color=colors[0],
                 linestyle='-',
                 label='labeled_loss')

        plt.plot(loss_dct['unlabeled_loss'],
                 color=colors[1],
                 linestyle='-',
                 label='unlabeled_loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        save_path = os.path.join(save_dir,
                                 runname + '_fold%d_test-retest_loss.png' % fold)
        plt.savefig(save_path)

        print(save_path, 'saved!')

    print('saving loss plots')
    plt.figure(figsize=(10, 7))
    plt.plot(loss_dct['train_loss'],
             color=colors[2],
             linestyle='-',
             label='train_loss')

    plt.plot(loss_dct['val_loss'],
             color=colors[3],
             linestyle='-',
             label='val_loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    save_path = os.path.join(save_dir, runname + '_fold%d_loss.png' % fold)
    plt.savefig(save_path)
    print(save_path, 'saved!')

def save_model(fold, model, optimizer, criterion, save_dir):
    """Function to save the trained model."""
    print("Saving final model...")

    save_filename = 'finalmodel-fold%d.pth'%fold
    save_path = os.path.join(save_dir, save_filename)

    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path)

    print(save_filename, 'saved!')
