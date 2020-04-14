import pathlib
import requests
import time
import json
import zipfile
import tqdm


zip_url = ""
dataset_path = pathlib.Path("datasets", "tdt4265")


class RequestWrapper:

    def __init__(self):
        self.server_url = "https://tdt4265-annotering.idi.ntnu.no"
        self.get_token()

    def get_token(self):
        group_number = input("group number:")
        try:
            group_number = int(group_number.strip())
        except Exception:
            print("Not a number. Write in your group number.")
            exit()
        user = f"group{group_number}"
        self.user = user
        password = input("Type in password (Given in the assignment PDF):")
        data = {"username": user, "password": password.strip(), "email": ""}
        r = requests.post(f"{self.server_url}/api/v1/auth/login", data=data)
        if r.status_code != 200:
            print("Could not authorize you.")
            print(r)
            exit()

        auth_string = "Token " + r.json()["key"]
        self.headers = {"Authorization": auth_string}
        print("Authentication OK.")

    def do_request(self, path):
        url = f"{self.server_url}{path}"
        r = requests.get(url, headers=self.headers)
        return r

    def download(self, path, target_path):
        url = f"{self.server_url}{path}"
        response = requests.get(url, stream=True, headers=self.headers)
        total_length = int(response.headers.get("content-length"))
        assert response.status_code == 200, \
            f"Did not download the images. Contact the TA. Status code: {response.status_code}"
        with open(target_path, "wb") as fp:
            for data in tqdm.tqdm(
                    response.iter_content(chunk_size=4096), total=total_length,
                    desc="Downloading images."):
                fp.write(data)


def download_labels(request_wrapper):
    status_code = 202
    print("Fetching labels.")
    while status_code == 202:
        response = request_wrapper.do_request("/api/v1/download/0/download")
        status_code = response.status_code
        if status_code == 202:
            print("The labes are being generated on the server - this could take up to 2 minutes.")
            print("Sleeping the project for 2 minutes - do not interrput the program")
            time.sleep(60*2)
            continue
        break
    if status_code != 200:
        print("Failure on download of dataset. Contact a TA.")
        exit()
    labels = json.loads(response.text)
    label_path = dataset_path.joinpath("train", "labels.json")
    with open(label_path, "w") as fp:
        json.dump(labels, fp)
    print("Labels downloaded.")


def make_image_symlinks(work_dataset, target_path):
    for subset in ["test", "train"]:
        image_source = work_dataset.joinpath(subset, "images")
        target_path.joinpath(subset, "images").symlink_to(
            image_source
        )


def download_images(request_wrapper):
    print("Extracting images.")
    work_dataset = pathlib.Path("/work", "datasets", "tdt4265")
    image_dir = dataset_path.joinpath("train", "images")
    to_delete = [image_dir, image_dir.parent.parent.joinpath("test", "images")]
    if image_dir.is_dir():
        print("The images already exists.")
        print(f"If there are any images missing, you can download again, but first you have to delete the directory {to_delete}")
        return
    if work_dataset.is_dir():
        print("You are working on a computer with the dataset under work_dataset.")
        print("We're going to copy all image files to your directory")
        make_image_symlinks(work_dataset, dataset_path)
        return
    zip_path = pathlib.Path("datasets", "tdt4265", "images_mini.zip")
    if not zip_path.is_file():
        print(f"Did not find the image zip file. Go to: {zip_url} and place it at the path: {zip_path}")
        exit()
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(dataset_path)

if __name__ == "__main__":
    # Download labels
    request_wrapper = RequestWrapper()
    dataset_path.joinpath("train").mkdir(exist_ok=True, parents=True)
    dataset_path.joinpath("test").mkdir(exist_ok=True, parents=True)
    download_images(request_wrapper)
    download_labels(request_wrapper)
    print("Dataset extracted to:", dataset_path)
