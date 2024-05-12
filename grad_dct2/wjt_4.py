import torch
import numpy as np
from PIL import Image
import multiprocessing

def M(n):
    return n // 2 if n % 2 == 0 else (n - 1) // 2

def binomial_terms(n, m, dtype=torch.float32, device=torch.device('cpu')):
    max_k = max(M(n), M(m)) + 1
    k_values = torch.arange(max_k, dtype=dtype, device=device)
    log_terms_n = torch.lgamma(torch.tensor([2 * n], dtype=dtype, device=device) - 2 * k_values + 1) - (torch.log(torch.tensor([2.0**n], dtype=dtype, device=device)) + torch.lgamma(k_values + 1) + torch.lgamma(torch.tensor([n], dtype=dtype, device=device) - k_values + 1) + torch.lgamma(torch.tensor([n], dtype=dtype, device=device)))
    log_terms_m = torch.lgamma(torch.tensor([2 * m], dtype=dtype, device=device) - 2 * k_values + 1) - (torch.log(torch.tensor([2.0**m], dtype=dtype, device=device)) + torch.lgamma(k_values + 1) + torch.lgamma(torch.tensor([m], dtype=dtype, device=device) - k_values + 1) + torch.lgamma(torch.tensor([m], dtype=dtype, device=device)))
    return log_terms_n, log_terms_m

def calculate_F_gpu(n, m, f, img_data, device):
    log_terms_n, log_terms_m = binomial_terms(n, m, dtype=torch.float32, device=device)
    log_F = torch.full((3,), float('-inf'), dtype=torch.float32, device=device)
    
    xy = torch.stack(torch.meshgrid(torch.arange(256, device=device), torch.arange(256, device=device)), dim=-1).float()
    xy += 1e-10  # Avoid division by zero
    
    img_data_tensor = torch.tensor(img_data, dtype=torch.float32, device=device)
    pixel_values = f(xy[..., 0], xy[..., 1], img_data_tensor)
    
    log_sum_y = torch.full((256, 3), float('-inf'), dtype=torch.float32, device=device)
    for k in range(M(m) + 1):
        exponent = log_terms_m[k] + (m - 2 * k) * torch.log(xy[..., 1])
        log_sum_y = torch.logaddexp(log_sum_y, exponent.unsqueeze(-1))

    log_sum_x = torch.full((256, 3), float('-inf'), dtype=torch.float32, device=device)
    for k in range(M(n) + 1):
        exponent = log_terms_n[k] + (n - 2 * k) * torch.log(xy[..., 0])
        log_sum_x = torch.logaddexp(log_sum_x, exponent.unsqueeze(-1))

    pixel_log = torch.log(torch.abs(pixel_values) + 1e-10)
    sum_exp = torch.logaddexp(log_F.unsqueeze(0).unsqueeze(0), pixel_log + log_sum_x.unsqueeze(0) + log_sum_y.unsqueeze(1))
    
    # Apply torch.max sequentially over dimensions 0 and 1
    max_over_dim0, _ = torch.max(sum_exp, dim=0)
    log_F, _ = torch.max(max_over_dim0, dim=0)

    return torch.exp(log_F).cpu().numpy()



def example_f(x, y, img_data):
    # Directly use tensor-based indexing. Ensure x and y are within the valid range [0, 255].
    x = torch.clamp(x.long(), 0, 255)
    y = torch.clamp(y.long(), 0, 255)
    
    # Use advanced indexing with tensors. This avoids the need for conversion to Python scalars.
    return img_data[x, y]

# Ensure the rest of your script is compatible with these changes.



def worker(device_id, start_n, end_n, img_data):
    device = torch.device(f'cuda:{device_id}')
    results = []
    for n in range(start_n, end_n):  # Adjust the range for each worker
        for m in range(256):
            F_values = calculate_F_gpu(n, m, example_f, img_data, device)
            results.append(f"F({n}, {m}) = R: {F_values[0]}, G: {F_values[1]}, B: {F_values[2]}\n")
    # You might want to save intermediate results from each worker to a file
    with open(f"results_{device_id}.txt", "w") as file:
        for result in results:
            file.write(result)

if __name__ == "__main__":
    img = Image.open("00001.png")
    img_data = np.array(img)

    num_gpus = torch.cuda.device_count()
    processes = []
    n_per_gpu = 256 // num_gpus

    # Set the start method for multiprocessing to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    for i in range(num_gpus):
        start_n = n_per_gpu * i
        end_n = n_per_gpu * (i + 1)
        p = multiprocessing.Process(target=worker, args=(i, start_n, end_n, img_data))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Results have been written to results files.")

