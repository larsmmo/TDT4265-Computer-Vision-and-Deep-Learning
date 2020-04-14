import pathlib
import zipfile

zip_url = "(zip URL not public yet. will be uploaded soon. Ask on piazza/ TA on mail: hakon.hukkelas@ntnu.no)"

if __name__ == "__main__":
    dataset_path = pathlib.Path("datasets", "waymo")
    if not dataset_path.parent.is_dir():
        dataset_path.parent.mkdir(exist_ok=True, parents=True)
    work_dataset_path = pathlib.Path("/work", "datasets", "waymo")
    if dataset_path.is_dir():
        print("Dataset already exists. If you want to download again, delete the folder", dataset_path.absolute())
        exit()
    if work_dataset_path.is_dir():
        print("You are working on a computer with the dataset under work_dataset.")
        print("We're going to copy all image files to your directory")
        print("Dataset setup finished. Extracted to:", dataset_path)

        dataset_path.symlink_to(work_dataset_path)
        exit()
    zip_path = dataset_path.parent.joinpath("waymo.zip")
    if not zip_path.is_file():
        print(f"Did not find the image zip file. Go to: {zip_url} and place it at the path: {zip_path}")
        exit()
    else:
        print("Found zip file. Extracting dataset")
        with zipfile.ZipFile(zip_path, "r") as fp:
            fp.extractall(dataset_path)
    print("Dataset setup finished. Extracted to:", dataset_path)
