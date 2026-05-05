import os

# Cosas a ignorar
IGNORE = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".DS_Store"
}

def print_tree(start_path=".", prefix=""):
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        print(prefix + "└── [Acceso denegado]")
        return

    # Filtrar lo que no quieres ver
    items = [item for item in items if item not in IGNORE]

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print(f"\nEstructura de carpetas desde: {os.getcwd()}\n")
    print_tree()