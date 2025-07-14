import torch
from torchvision.transforms import v2
from PIL import Image
import os
import pandas as pd
import functools
import multiprocessing as mp
import re

def convert_images_to_tensor(image_paths, desired_height, desired_width):
    """
    Convert a list of images to a single tensor with specified dimensions.

    Args:
        image_paths (list): List of paths to the images
        desired_height (int): Desired height for the output tensor
        desired_width (int): Desired width for the output tensor

    Returns:
        torch.Tensor: Tensor of shape (C, H, W) where C is the number of channels,
                     H is desired_height, and W is desired_width
    """
    # Define the transformation pipeline
    transform = v2.Compose([
        v2.Resize((desired_height, desired_width), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.uint8, scale=True)
    ])

    # Process each image and stack them into a tensor
    tensors = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            tensor = transform(img)
            tensors.append(tensor)

    # Stack all tensors along the channel dimension 
    if tensors:
        result = torch.cat(tensors, dim=0)
        return result
    else:
        return torch.empty(0)
    
def write_output_tensor(input, output_folder, desired_height, desired_width):
    paths, name = input
    new_path = os.path.join(output_folder, name+".pt")
    if os.path.isfile(new_path):
        return None
    try:
        tensor = convert_images_to_tensor(paths, desired_height, desired_width)
    except Exception as e:
        return None
    torch.save(tensor, os.path.join(output_folder, name+".pt"))

    
def process_row_batch(frow,output_folder, desired_height, desired_width):
    #building the name of the tensor
    path_tensor = os.path.join(output_folder,frow["Metadata_Plate"]+"_"+frow["Metadata_Well"]+"_"+frow["Metadata_Window"]+".pt")
    if os.path.exists(path_tensor):
        return None
    # Extracting iamge path
    images_path = frow[[x for x in frow.index if x.startswith("Image")]].tolist()
    tensor = convert_images_to_tensor(images_path,desired_height, desired_width)

    # Saving the tensor
    torch.save(tensor, path_tensor)

def create_tensors_from_metadata(pmetas,output_folder, desired_height, desired_width, nmax=None):
    print("Processing {}".format(pmetas))
    # Preprocessing metas data files
    metas = pd.read_csv(pmetas)
    # print("Processing {}".format(pmetas))
    images_col = [x for x in metas.columns if x.startswith("Image")]
    # Changing the files
    for col in images_col:
        # Adapt the from windows to wsl
        metas[col] = metas[col].str.replace("\\", "/").str.replace("C:", "/mnt/c")
    

    irow = 0
    for _,frow in metas.iterrows():
        if nmax is not None and irow>=nmax:
            return None
        _ = process_row_batch(frow,output_folder,desired_height, desired_width)  
        irow += 1     
    return None         

if __name__=="__main__":
    FOLDERS = [
        "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/GR00003300/Images",
        "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/GR00003310/Images",
        "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/GR00003340/Images",
        "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/GR00004371/Images"       
    ]

    OUTPUT_FOLDER_64 = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/resized_tensor_64_uint8"
    OUTPUT_FOLDER_32 = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/resized_tensor_32_uint8"

    def get_tiff_path(folder):
        return [os.path.join(folder,x) for x in os.listdir(folder) if x.endswith(".tiff")]

    all_paths = [y for f in FOLDERS for y in get_tiff_path(f)]

    info_regex = ".*(GR[0123456789]{8})\/Images\/(r[cfp0123456789]+)\-ch([1-5])"
    # info_regex = "(r[cfp0123456789]+)\-ch([1-5])"
    def extract_info(path):
        m = re.match(info_regex,path)
        if m is None:
            return path, None, None, None
        return path, m.group(1)+m.group(2), m.group(3)
    
    infos = [extract_info(x) for x in all_paths]

    df_infos = pd.DataFrame(infos,columns=["path","frame","channel"])

    df_infos = df_infos.sort_values(["frame","channel"])

    valid_channels = df_infos.groupby("frame")["channel"].apply(lambda x:"|".join(x))
    valid_channels = valid_channels[valid_channels=="1|2|3|4|5"]
    valid_channels = valid_channels.index.tolist()

    df_infos = df_infos[df_infos["frame"].isin(valid_channels)]

    # Chunk values by 5
    vpaths = df_infos.path.tolist()
    vpaths = [vpaths[i:i+5] for i in range(0,len(vpaths),5)]


    if not os.path.isdir(OUTPUT_FOLDER_64):
        os.makedirs(OUTPUT_FOLDER_64)
    if not os.path.isdir(OUTPUT_FOLDER_32):
        os.makedirs(OUTPUT_FOLDER_32)

    vtuples = list(zip(vpaths,valid_channels))

    pfun64 = functools.partial(write_output_tensor, desired_height=64, desired_width=64, output_folder=OUTPUT_FOLDER_64)
    pfun32 = functools.partial(write_output_tensor, desired_height=32, desired_width=32, output_folder=OUTPUT_FOLDER_32)

    print("Writing 64px images")
    with mp.Pool(processes=9) as pool:
        pool.map(pfun64, vtuples)
    print("Done")

    print("Writing 32px images")
    with mp.Pool(processes=9) as pool:
        pool.map(pfun32, vtuples)
    print("Done")


