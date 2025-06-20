import torch
from torchvision.transforms import v2
from PIL import Image
import os
import pandas as pd
import functools
import multiprocessing as mp

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

    FOLDER_BATCHES = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/results_target_docker"
    OUTPUT_FOLDER = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/resized_tensor_128_uint8"

    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


    RESIZED_HEIGHT = 128
    RESIZED_WIDTH = 128

    # Partial procesing function
    proc_fun = functools.partial(create_tensors_from_metadata,output_folder=OUTPUT_FOLDER, desired_height=RESIZED_HEIGHT, desired_width=RESIZED_WIDTH)

    # We list all the metadata in batches 
    abatches = [os.path.join(FOLDER_BATCHES, batch) for batch in os.listdir(FOLDER_BATCHES) if os.path.exists(os.path.join(FOLDER_BATCHES,batch,"measures"))]

    def get_metadata_file(folder):
        lfiles = [x for x in os.listdir(folder) if x.startswith("metadata") ]
        return os.path.join(folder,lfiles[0])

    print("Processing batches...")

    met_files = [get_metadata_file(folder) for folder in abatches]
    # met_files = met_files[:10]

    # proc_fun(met_files[0])

    with mp.Pool(processes=5) as pool:
        pool.map(proc_fun, met_files)
