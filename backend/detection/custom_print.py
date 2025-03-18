import builtins  # Import built-in functions

# Save the original print function
original_print = builtins.print


def custom_print(*args, **kwargs):
    """Custom print function that writes to both console and a file."""
    with open("output.txt", "a") as f:
        message = " ".join(str(arg) for arg in args)  # Convert arguments to a string
        f.write(message + "\n")  # Write to file

    original_print(*args, **kwargs)  # Also print to console


# Override the global print function
builtins.print = custom_print
