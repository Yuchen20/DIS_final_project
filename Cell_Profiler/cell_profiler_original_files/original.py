import os
import shutil

def move_target_npy_files(src_folder, dst_folder):
    """
    Move all files ending with '_target.npy' from src_folder to dst_folder.
    """
    if not os.path.isdir(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if fname.endswith('_target.npy'):
            src_path = os.path.join(src_folder, fname)
            new_fname = fname.replace('_target.npy', '_pred.npy')
            dst_path = os.path.join(dst_folder, new_fname)
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)
            print(f"Copied, renamed, and removed original: {src_path} -> {dst_path}")

if __name__ == "__main__":
    src_folder = "Cell_Profiler/cell_profiler_original_files/unet"
    dst_folder = "Cell_Profiler/cell_profiler_original_files/real"
    move_target_npy_files(src_folder, dst_folder)
    
# Example usage:
# move_target_npy_files("Cell_Profiler/cell_profiler_original_files/unet", "Cell_Profiler/cell_profiler_original_files/targets")
