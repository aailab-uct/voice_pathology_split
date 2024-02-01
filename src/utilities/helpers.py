"""
Modul with various helper-functions.
"""
from pathlib import Path


def clear_folder(folder_path: Path):
    """
    Deletes all files in given folder_path
    :param folder_path: Path to folder where all files will be deleted
    :return: None
    """
    files_to_delete = folder_path.glob("*")
    for file in files_to_delete:
        file.unlink()
