import torch
import torch.multiprocessing as mp
from PIL import Image
import numpy as np
import math

def M(n):
    return n // 2 if n % 2 == 0 else (n - 1) // 2

def log_stirling(n):
    if n == 0 or n == 1:
        return 0
    return 0.5 * math.log(2 * math.pi * n,100) + n * math.log(n / math.e,100)

binomial_log_cache = {}

def binomial_term_log(n, k):
    if (n, k) in binomial_log_cache:
        return binomial_log_cache[(n, k)]
    try:
        log_term = math.log(abs((-1)**k),100) + log_stirling(2 * n - 2 * k) - \
                   (n * math.log(2,100) + log_stirling(k) + log_stirling(n - k) + log_stirling(n - 2))
        binomial_log_cache[(n, k)] = log_term
        return log_term
    except ValueError:
        return -float('inf')

def add_logs(log_a, log_b):
    if log_a == -float('inf'):
        return log_b
    elif log_b == -float('inf'):
        return log_a
    if log_b > log_a:
        log_a, log_b = log_b, log_a
    return log_a + math.log(1 + 10 ** ((log_b - log_a) * 2), 100) / 2

def calculate_F(n, m, f, image_tensor):
    log_F_sum_r = log_F_sum_g = log_F_sum_b = -float('inf')
    
    for y in range(1, 257):
        log_sum_y = -float('inf')
        for k in range(M(m) + 1):
            log_term = binomial_term_log(m, k) + (m - 2 * k) * math.log(y,100)
            log_sum_y = add_logs(log_sum_y, log_term)

        for x in range(1, 257):
            log_sum_x = -float('inf')
            for k in range(M(n) + 1):
                log_term = binomial_term_log(n, k) + (n - 2 * k) * math.log(x,100)
                log_sum_x = add_logs(log_sum_x, log_term)

            r, g, b = f(x, y, image_tensor)
            if r > 0:
                log_term_r = math.log(r,100) + log_sum_x + log_sum_y
                log_F_sum_r = add_logs(log_F_sum_r, log_term_r)
            if g > 0:
                log_term_g = math.log(g,100) + log_sum_x + log_sum_y
                log_F_sum_g = add_logs(log_F_sum_g, log_term_g)
            if b > 0:
                log_term_b = math.log(b,100) + log_sum_x + log_sum_y
                log_F_sum_b = add_logs(log_F_sum_b, log_term_b)

    return log_F_sum_r, log_F_sum_g, log_F_sum_b

def example_f(x, y, image_tensor):
    r, g, b = image_tensor[y-1, x-1]
    return r.item(), g.item(), b.item()

def worker(gpu, n_range, m_range, results):
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    print(f"Using GPU: {gpu}")

    image_path = "00001.png"
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_data = np.array(image)
    image_tensor = torch.from_numpy(image_data).to(device)

    for n in n_range:
        for m in m_range:
            log_result_r, log_result_g, log_result_b = calculate_F(n, m, example_f, image_tensor)
            results.append(f"F({n}, {m}) RGB Log = {log_result_r}, {log_result_g}, {log_result_b}")

def main():
    num_gpus = torch.cuda.device_count()
    mp.set_start_method('spawn')

    manager = mp.Manager()
    results = manager.list()

    processes = []
    for gpu in range(num_gpus):
        n_range = range(1 + gpu * 64, 1 + (gpu + 1) * 64)
        m_range = range(1, 257)
        p = mp.Process(target=worker, args=(gpu, n_range, m_range, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open("god.txt", "w") as file:
        for result in results:
            file.write(result + "\n")

if __name__ == "__main__":
    main()
