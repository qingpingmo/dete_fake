from PIL import Image
import math

def M(n):
    """Calculate M based on n."""
    return n // 2 if n % 2 == 0 else (n - 1) // 2

def binomial_term(n, k):
    """Calculate the binomial term of the formula."""
    try:
        # Calculate the logarithm of the term to avoid overflow
        log_term = math.log((-1)**k) + math.log(math.factorial(2 * n - 2 * k)) - math.log(2**n) - math.log(math.factorial(k)) - math.log(math.factorial(n - k)) - math.log(math.factorial(n - 2))
        return log_term
    except ValueError:
        return float('-inf')  # Return negative infinity for invalid calculations

def calculate_F(n, m, f, img_data):
    """Calculate the F(n, m) based on the given formula."""
    log_F = [float('-inf'), float('-inf'), float('-inf')]  # Use negative infinity to represent log(0) for RGB channels
    for y in range(256):
        log_sum_y = [float('-inf'), float('-inf'), float('-inf')]  # For RGB channels
        for k in range(M(m) + 1):
            log_term = binomial_term(m, k)
            for channel in range(3):
                if y != 0:
                    exponent = log_term + (m - 2 * k) * math.log(abs(y))
                    if exponent != float('-inf'):
                        # Avoid overflow by using math.log1p for numerical stability
                        log_sum_y[channel] = log_sum_y[channel] + math.log1p(math.exp(exponent - log_sum_y[channel]))

        for x in range(256):
            log_sum_x = [float('-inf'), float('-inf'), float('-inf')]  # For RGB channels
            for k in range(M(n) + 1):
                log_term = binomial_term(n, k)
                for channel in range(3):
                    if x != 0:
                        exponent = log_term + (n - 2 * k) * math.log(abs(x))
                        if exponent != float('-inf'):
                            # Avoibd overflow by using math.log1p for numerical stability
                            log_sum_x[channel] = log_sum_x[channel] + math.log1p(math.exp(exponent - log_sum_x[channel]))

            pixel = f(x, y, img_data)
            for channel in range(3):
                pixel_value = abs(pixel[channel]) if pixel[channel] != 0 else 1e-10
                # Avoid overflow by using math.log1p for numerical stability
                sum_exp = log_F[channel] + math.log1p(math.exp(math.log(pixel_value) + log_sum_x[channel] + log_sum_y[channel] - log_F[channel]))
                if sum_exp > 0:
                    log_F[channel] = sum_exp

    return [math.exp(log_F[channel]) if log_F[channel] != float('-inf') else 0 for channel in range(3)]





def example_f(x, y, img_data):
    """Return the RGB values of the pixel at (x, y) in the image."""
    return img_data[x, y]  # Get pixel value at (x, y)

if __name__ == "__main__":
    # Load the image
    img = Image.open("00001.png")
    img_data = img.load()

    # Open a text file for writing the results
    with open("results.txt", "w") as file:
        for n in range(256):
            for m in range(256):
                F_values = calculate_F(n, m, example_f, img_data)
                file.write(f"F({n}, {m}) = R: {F_values[0]}, G: {F_values[1]}, B: {F_values[2]}\n")

    print("Results have been written to results.txt")
