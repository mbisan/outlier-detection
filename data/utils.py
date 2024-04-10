import os

def walk_path(root_dir, extension=".jpg"):
    return [
        os.path.join(looproot, filename) for looproot, _, filenames in os.walk(root_dir)
            for filename in filenames if
            (filename.endswith(extension) and os.path.isfile(os.path.join(looproot, filename)))
        ]
