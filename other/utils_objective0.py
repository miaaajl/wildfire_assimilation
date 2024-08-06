import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def make_gif(wildfire, gif_path):
    # Convert each image to a PIL Image object, scaling values from 0-1 to 0-255
    pil_images = [Image.fromarray(image * 255).convert('L') for image in wildfire]
    # Save as a GIF
    pil_images[0].save(
        gif_path,
        format='GIF',
        save_all=True,
        append_images=pil_images[1:],  # Append the rest of the images
        duration=10,  # Duration in milliseconds
        loop=0  # Loop count; 
    )

def plot_comparison(training_set, testing_set):
    """
    Plots the comparison of burned pixels in the training and testing sets.
    """
    # Calculate the number of burned pixels in each wildfire
    df_training = {f'wildfire_{i}': (training_set[i][-1].sum() / (255 * 255)) for i in range(len(training_set))}
    df_testing = {f'wildfire_{i}': (testing_set[i][-1].sum() / (255 * 255)) for i in range(len(testing_set))}
    # Prepare data for plotting
    training_values = list(df_training.values())
    testing_values = list(df_testing.values())

    print(len(training_values))
    print(len(testing_values))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot([i for i in range(len(training_values))], training_values, 'bx', label='Training Set')

    # Plot testing data
    plt.plot([int(125/50*i) for i in range(len(testing_values))], testing_values, 'ro', label='Testing Set')

    # Add labels and title
    plt.xlabel('Wildfires index of Training Set')
    plt.ylabel('Burned Pixels')
    plt.title('Comparison of Burned Pixels in Training and Testing Sets')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_input_and_target(inputs, output, training=True):
    """
    Plots the input images and the output image from the DataLoader.

    Parameters:
    inputs (torch.Tensor): The input images, should have shape (N, 256, 256) or (N, 1, 256, 256).
    output (torch.Tensor): The output image, should have shape (1, 256, 256).
    """
    # Convert tensors to numpy arrays
    inputs_np = inputs.cpu().numpy()
    output_np = output.cpu().numpy()

    # Ensure inputs have the correct shape
    if inputs_np.ndim == 3:
        # If inputs are (N, 256, 256), add a channel dimension to match (N, 1, 256, 256)
        inputs_np = inputs_np[:, np.newaxis, ...]

    # Remove the channel dimension from output if it's (1, 256, 256)
    if output_np.shape[0] == 1:
        output_np = output_np.squeeze(0)

    num_inputs = inputs_np.shape[0]

    # Create subplots: number of inputs + 1 for the output image
    fig, axes = plt.subplots(1, num_inputs + 1, figsize=(15, 5))

    for i in range(num_inputs):
        ax = axes[i]
        ax.imshow(inputs_np[i, 0], cmap='gray')  # Use the first channel for visualization
        ax.set_title(f'Input {i + 1}')
        ax.axis('off')

    # Plot the output image
    ax = axes[num_inputs]
    ax.imshow(output_np, cmap='gray')
    ax.set_title('Output')
    ax.axis('off')

    if training:
        fig.suptitle('Visualization of Elements in the Training Batch', fontsize=16)
    else:
        fig.suptitle('Visualization of Elements in the Validation Batch', fontsize=16)

    plt.tight_layout()
    plt.show()

