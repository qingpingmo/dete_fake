import math


def F(n, m, f):
    def coeff(z, k):
        return ((-1) ** k) * math.factorial(2 * z - 2 * k) / (2 ** z * math.factorial(k) * math.factorial(z - k) * math.factorial(z - 2))

    def M(z):
        return z // 2 if z % 2 == 0 else (z - 1) // 2
    
    total_sum = 0
    for y in range(256):
        sum_y = sum([coeff(m, k) * (y ** (m - 2 * k)) for k in range(M(m) + 1)])
        for x in range(256):
            sum_x = sum([coeff(n, k) * (x ** (n - 2 * k)) for k in range(M(n) + 1)])
            total_sum += f(x, y) * sum_x * sum_y
    return total_sum

# Example usage with a dummy f(x, y) function
def f(x, y):
    # Simple example function, replace with actual f(x, y)
    return x+y

n = 2  # Example value
m = 3  # Example value

F_result = F(n, m, f)
print(F_result)




