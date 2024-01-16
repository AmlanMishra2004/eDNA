import os

# Define the directory and file timestamp
dir1 = 'old_best_saved_models'
dir2 = 'saved_models'
timestamp = '20240103_210217'

file_name = 'best_model_' + timestamp + '.pt'
file_path1 = os.path.join(dir1, file_name)
file_path2 = os.path.join(dir2, file_name)

if os.path.exists(file_path1):
    print(f'The file EXISTS in old_best_saved_models: {file_path1}')
else:
    print(f'The file DOES NOT exist in old_best_saved_models: {file_path1}')

if os.path.exists(file_path2):
    print(f'The file EXISTS in saved_models: {file_path2}')
else:
    print(f'The file DOES NOT exist in saved_models: {file_path2}')


