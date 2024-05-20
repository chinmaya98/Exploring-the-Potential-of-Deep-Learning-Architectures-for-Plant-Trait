import matplotlib.pyplot as plt

class visualization():
    def mae_plot(history):
        """
        Input: History of the model built+fitted
        Output: N/A
        Function: Displays the loss curve for the model built
        """
        plt.figure(figsize=(10, 7))
        plt.plot(history.history['loss'][5:], label='Train Loss')  # Start from the sixth epoch
        plt.plot(history.history['val_loss'][5:], label='Validation Loss')  # Start from the sixth epoch
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)  # Adds grid to the plot
        plt.show()

