import tensorflow as tf

class setupGPUS():
    def check_GPUS():
        """
        Input: N/A
        Output: N/A
        Function: Checks the no. of GPUs available in the system
        """
        # Retrieve a list of GPU devices
        available_gpus = tf.config.list_physical_devices('GPU')

        # Check if any GPUs are available
        if available_gpus:
            try:
                # Enable memory growth setting for each GPU
                for gpu in available_gpus:
                    tf.config.experimental.set_memory_growth(gpu, enable=True)
                # After setting memory growth, get the count of accessible logical GPUs
                accessible_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(available_gpus)} Physical GPUs, {len(accessible_gpus)} Logical GPUs")
            except RuntimeError as error:
                # Error handling for cases where memory growth could not be set (typically because GPUs are already in use)
                print(f"Error during setting GPU configuration: {error}")


