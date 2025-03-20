import matplotlib.pyplot as plt
import numpy as np


def kvadraticka_funkce(x):
    y = x * x
    return y


if __name__ == "__main__":
    # Generate x values
    x = np.linspace(-10, 10, 100)

    # Calculate corresponding y values
    y = kvadraticka_funkce(x)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='y = x^2')
    plt.title('Kvadratická funkce: y = x^2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Add the origin point
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Show the plot
    plt.show()