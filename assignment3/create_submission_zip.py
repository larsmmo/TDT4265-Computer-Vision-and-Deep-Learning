import os
import zipfile

# If you create other files, edit this list to include them in the .zip file.
files_to_include = [
    "task2.py",
    "task2_train.ipynb",
    "task4b.py",
    "utils.py",
    "dataloaders.py"
]

for filepath in files_to_include:
    if not os.path.isfile(filepath):
        print("Did not find file: {}".format(filepath))
        print("Terminating program without creating zipfile.")
        exit(0)

zipfile_path = "assignment_code.zip"


print("-"*80)
with zipfile.ZipFile(zipfile_path, "w") as fp:
    for filepath in files_to_include:
        fp.write(filepath)
        print("Adding file:", filepath)
print("-"*80)
print("Zipfile saved to: {}".format(zipfile_path))
print("Please, upload your assignment PDF file outside the zipfile to blackboard.")
