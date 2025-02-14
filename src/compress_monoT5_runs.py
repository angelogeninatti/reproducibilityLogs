import os
import tarfile


def create_tar_gz_for_each_file(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return
    for filename in os.listdir(directory):
        if filename == '.DS_Store':
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            out_tarfile = os.path.splitext(file_path)[0] + ".tar.gz"
            with tarfile.open(out_tarfile, "w:gz") as tar:
                tar.add(file_path, arcname=os.path.basename(file_path))
            print(f"Created tarball: {out_tarfile}")


def main():
    directory_path = './runs/monoT5/txt'
    create_tar_gz_for_each_file(directory_path)


if __name__ == '__main__':
    main()
