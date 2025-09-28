import os

def check_existing_file(file_path):
    """
    Check if a file exists.
    Returns True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def check_existing_folder(folder_path):
    """
    Check if a directory (folder) exists.
    Returns True if the directory exists, False otherwise.
    """
    return os.path.isdir(folder_path)
