import os
import zipfile

files_to_include = [
    "task2.py",
    "task2a.py",
    "task4.py",
    "task4a.py",
    "utils.py",
    "mnist.py"
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
