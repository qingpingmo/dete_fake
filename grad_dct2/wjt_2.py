import math

def M(n):
    """Calculate M based on n."""
    return n // 2 if n % 2 == 0 else (n - 1) // 2

def binomial_term(n, k):
    """Calculate the binomial term of the formula."""
    try:
        term = ((-1)**k * math.factorial(2 * n - 2 * k)) / (2**n * math.factorial(k) * math.factorial(n - k) * math.factorial(n - 2))
        return term
    except ValueError:
        return 0  # In case of invalid factorial calculation

def calculate_F(n, m, f):
    """Calculate the F(n, m) based on the given formula."""
    F = 0
    for y in range(256):
        sum_y = 0
        for k in range(M(m) + 1):
            sum_y += binomial_term(m, k) * (y ** (m - 2 * k))
        
        for x in range(256):
            sum_x = 0
            for k in range(M(n) + 1):
                sum_x += binomial_term(n, k) * (x ** (n - 2 * k))
            
            F += f(x, y) * sum_x * sum_y
    
    return F

# Example usage
def example_f(x, y):
    """Example f(x, y) function."""
    return x + y  # Replace this with the actual function

if __name__ == "__main__":
    n = int(input("Enter n: "))
    m = int(input("Enter m: "))
    result = calculate_F(n, m, example_f)
    print(f"F({n}, {m}) =", result)
