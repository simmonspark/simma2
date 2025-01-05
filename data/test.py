import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Load the tensor file
def load_tensor_file(file_path):
    return torch.load(file_path)

# Save tensor slice function with input_ids and labels
def save_tensor_slice(slice_tensor, index, output_dir, batch_size):
    try:
        # Ensure slice_tensor has the shape (batch_size, ids_per_batch)
        if slice_tensor.dim() != 2 or slice_tensor.size(0) != batch_size:
            raise ValueError(
                f"Slice tensor at index {index} has shape {slice_tensor.shape}, expected ({batch_size}, ids_per_batch)"
            )

        # Convert tensor to NumPy array
        input_ids = slice_tensor.cpu().numpy()  # Shape: (batch_size, ids_per_batch)

        # Shift input_ids by one to create labels
        labels = np.roll(input_ids, -1, axis=1)

        # Set the last token in each sequence to a special token (e.g., 0) or ignore index
        labels[:, -1] = 0  # 변경 가능: 패딩 토큰이나 무시할 인덱스로 설정

        # Define the file name and path
        file_name = f"tensor_{index}.npz"  # .npz 형식으로 변경
        file_path = os.path.join(output_dir, file_name)

        # Save using np.savez (no pickle needed)
        np.savez(file_path, input_ids=input_ids, labels=labels)
        print(f"Saved tensor_{index}.npz to {file_path}")
    except Exception as e:
        print(f"Error saving tensor_{index}: {e}")

# Main processing
def main():
    TENSOR_FILE = "encoded_context.pt"
    output_dir = "/media/sien/media/data/train_data/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Load the tensor
    loaded_data = load_tensor_file(TENSOR_FILE)

    # Check if 'input_ids' exists in the loaded data
    if 'input_ids' not in loaded_data:
        print(f"'input_ids' not found in {TENSOR_FILE}")
        return

    input_ids = loaded_data['input_ids']  # Assuming shape is (1, n_dim)

    # Ensure input_ids is 2D
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # Reshape to (1, total_length)

    # Define the slice size
    batch_size = 2  # 배치 크기를 3으로 설정
    ids_per_batch = 2048  # 원하는 시퀀스 길이로 설정 (예: 2048)

    # Calculate total elements per sequence
    num_sequences, total_length = input_ids.shape  # (num_sequences, total_length)

    # Adjust num_sequences to match batch_size
    if num_sequences < batch_size:
        # 반복하여 배치 크기를 맞춤
        repeat_factor = (batch_size + num_sequences - 1) // num_sequences
        input_ids = input_ids.repeat(repeat_factor, 1)
        num_sequences = input_ids.shape[0]

    # Now, input_ids has at least 'batch_size' sequences
    input_ids = input_ids[:batch_size, :]  # Shape: (batch_size, total_length)

    # Calculate the number of complete slices per sequence
    num_complete_slices = total_length // ids_per_batch

    # Calculate the total number of complete slices
    total_complete_slices = num_complete_slices

    # Slice the tensor to fit into complete (batch_size, ids_per_batch) batches
    complete_tensor = input_ids[:, :num_complete_slices * ids_per_batch]  # Shape: (batch_size, num_complete_slices * ids_per_batch)

    # Reshape the complete part of the tensor to (batch_size, num_complete_slices, ids_per_batch)
    reshaped_tensor = complete_tensor.view(batch_size, num_complete_slices, ids_per_batch)  # Shape: (batch_size, num_complete_slices, ids_per_batch)

    # Iterate over each slice and save them
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(num_complete_slices):
            slice_tensor = reshaped_tensor[:, i, :]  # Shape: (batch_size, ids_per_batch)
            futures.append(executor.submit(save_tensor_slice, slice_tensor, i, output_dir, batch_size))

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in future: {e}")

    # Handle and save remaining elements if any
    remaining_elements = total_length - num_complete_slices * ids_per_batch
    if remaining_elements > 0:
        remaining_tensor = input_ids[:, num_complete_slices * ids_per_batch:]

        # Check if remaining_tensor is non-empty
        if remaining_tensor.size(1) > 0:
            # Shift remaining_tensor to create labels
            labels = np.roll(remaining_tensor.cpu().numpy(), -1, axis=1)
            labels[:, -1] = 0  # 변경 가능: 패딩 토큰이나 무시할 인덱스로 설정

            # Create a dictionary containing both input_ids and labels
            sample = {
                'input_ids': remaining_tensor.cpu().numpy(),
                'labels': labels
            }

            # Define the file name and path for remaining tensor
            file_name = f"tensor_remaining.npz"
            file_path = os.path.join(output_dir, file_name)

            # Save using np.savez
            try:
                np.savez(file_path, input_ids=sample['input_ids'], labels=sample['labels'])
                print(f"Saved remaining tensor to {file_path}")
            except Exception as e:
                print(f"Error saving remaining tensor: {e}")
        else:
            print("No remaining elements to save.")
    else:
        print("No remaining elements to save.")

if __name__ == "__main__":
    main()
