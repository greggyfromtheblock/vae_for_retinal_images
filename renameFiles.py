import os


def rename(jpg_list):
    numbers = [str(i) for i in range(10)]
    for jpg in jpg_list:
        for i in range(1, 4):

            if jpg[i] not in numbers:
                new_filename = "0" * (4 - i) + jpg
                os.rename(jpg, new_filename)
                break

    print("Renaming succesfully")


path = "/home/henrik/PycharmProjects/Project A - VAE Retina/ODIR-5K_Training_Images/"
odir = "ODIR-5K_Training_Dataset"

# Change directory
os.chdir(path + odir)
jpg_list = os.listdir().copy()
rename(jpg_list)
jpg_list.sort()
print(jpg_list)