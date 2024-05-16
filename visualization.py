import matplotlib.pyplot as plt

class visualization():
    def mae_plot(history):
        plt.figure(figsize=(10, 7))
        plt.plot(history.history['loss'][5:], label='Train Loss')  # Start from the sixth epoch
        plt.plot(history.history['val_loss'][5:], label='Validation Loss')  # Start from the sixth epoch
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)  # Adds grid to the plot
        plt.show()

    def r2_plot(history):
        plt.figure(figsize=(10, 7))
        plt.plot(history.history['r2'][5:], label='R2')  # Start from the sixth epoch
        plt.plot(history.history['val_r2'][5:], label='Validation R2')  # Start from the sixth epoch
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.title('Training and Validation R2 Score')
        plt.legend()
        plt.grid(True)  # Adds grid to the plot
        plt.show()



