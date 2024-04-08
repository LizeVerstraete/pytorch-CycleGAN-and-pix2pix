import sys
from PIL import Image

A_paths = [r'/esat/biomeddata/kkontras/r0786880/biopsy_data_filtered/Microbiome CRC/B-1863984_01-04_HE_20221018/B-1863984_01-04_HE_2022101881609_101808.png',r'/esat/biomeddata/kkontras/r0786880/biopsy_data_filtered/Microbiome CRC/B-1863984_01-04_HE_20221018/B-1863984_01-04_HE_2022101881609_104280.png' ]

def getitem(index):

    A_path = A_paths[index]  # make sure index is within then range

    try:
        A = Image.open(A_path).convert('RGB')
    except OSError as e:
        print(f"Error: Failed to open or process the image '{A_path}': {e}", file=sys.stderr)
        index -= 1
        return getitem(index)

    # apply image transformation
    return {'A': A,'A_paths': A_path}

getitem(0)
getitem(1)