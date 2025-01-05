import torch
from first_stage_test import load_tensor_file
# Load the tensor file

TENSOR_FILE = "/media/sien/media/code/simma2/data/encoded_context.pt"
input_ids = load_tensor_file(TENSOR_FILE)

# Define the slice size (4, 8047)
num_rows = 4
num_cols = 8047

# Calculate the total number of slices
total_elements = input_ids.numel()
tensor_slices = total_elements // (num_rows * num_cols)

# Reshape the input tensor
reshaped_tensor = input_ids.view(-1, num_rows, num_cols)

# Save each slice to a separate file
output_dir = "/media/sien/media/code/simma2/data/"
for i in range(tensor_slices):
    file_name = f"tensor_{i}.pt"
    file_path = output_dir + file_name
    torch.save(reshaped_tensor[i], file_path)
    print(f"Saved tensor_{i} to {file_path}")

# Handle any remaining elements if the total size is not divisible
remaining_elements = total_elements % (num_rows * num_cols)
if remaining_elements > 0:
    remaining_tensor = input_ids[-remaining_elements:].view(1, -1)
    file_name = f"tensor_remaining.pt"
    file_path = output_dir + file_name
    torch.save(remaining_tensor, file_path)
    print(f"Saved remaining tensor to {file_path}")
