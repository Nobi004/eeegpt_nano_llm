import os

project_root = 'eee_gpt'
dirs = [
    os.path.join(project_root, 'data', 'raw'),
    os.path.join(project_root, 'data', 'processed'),
    os.path.join(project_root, 'src'),
    os.path.join(project_root, 'models'),
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

print("Directory structure created.")