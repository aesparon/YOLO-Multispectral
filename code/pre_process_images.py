import os
import glob
import numpy as np
import shutil
import rasterio
#from rasterio.merge import merge
#from rasterio.enums import Resampling
#from PIL import Image
import glob
import tifffile
import numpy as np
from pathlib import Path
from importlib.util import find_spec



def stack_rgb_geotiff(red_path, green_path, blue_path, output_path):
    with rasterio.open(red_path) as red_src, \
         rasterio.open(green_path) as green_src, \
         rasterio.open(blue_path) as blue_src:

        # Read 16-bit data
        red = red_src.read(1)
        green = green_src.read(1)
        blue = blue_src.read(1)

        # Check size consistency
        if red.shape != green.shape or red.shape != blue.shape:
            raise ValueError(f"Shape mismatch: {os.path.basename(red_path)}")

        # Stack bands into 3-band array
        rgb_stack = np.stack([red, green, blue], axis=0)

        # Copy metadata from red band
        meta = red_src.meta.copy()
        meta.update({
            "count": 3,
            "dtype": 'uint16',
            "driver": 'GTiff'
        })

        # Save multiband GeoTIFF
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(rgb_stack)
        print(f"✅ Saved: {output_path}")

def batch_stack_rgb_triplets(input_folder, output_folder , post_append = ''):
    os.makedirs(output_folder, exist_ok=True)

    # Find all red band files
    red_files = glob.glob(os.path.join(input_folder, "*_R.png"))
    red_files.sort()

    for red_path in red_files:
        base_name = os.path.basename(red_path).replace("_R.png", "")
        green_path = os.path.join(input_folder, base_name + "_G.png")
        blue_path = os.path.join(input_folder, base_name + "_B.png")

        # Check if corresponding green and blue images exist
        if not (os.path.exists(green_path) and os.path.exists(blue_path)):
            print(f"⚠️ Skipping {base_name}: Missing green or blue channel")
            continue

        # check later png to tif
        output_path = os.path.join(output_folder, base_name + post_append + ".tif")
        try:
            stack_rgb_geotiff(red_path, green_path, blue_path, output_path)
        except Exception as e:
            print(f"❌ Failed for {base_name}: {e}")











def stack_5band_geotiff(r_path, g_path, b_path, nir_path, re_path, output_path):
    # Open all bands
    with rasterio.open(r_path) as r_src, \
         rasterio.open(g_path) as g_src, \
         rasterio.open(b_path) as b_src, \
         rasterio.open(nir_path) as nir_src, \
         rasterio.open(re_path) as re_src:

        # Read data
        r = r_src.read(1)
        g = g_src.read(1)
        b = b_src.read(1)
        nir = nir_src.read(1)
        re = re_src.read(1)

        # Validate shape consistency
        if not (r.shape == g.shape == b.shape == nir.shape == re.shape):
            raise ValueError("All bands must have the same shape")

        # Stack into 5-band array
        stacked = np.stack([r, g, b, nir, re], axis=0)

        # Copy metadata from R band
        meta = r_src.meta.copy()
        meta.update({
            "count": 5,
            "dtype": "uint16",
            "driver": "GTiff"
        })


        # Write output GeoTIFF
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(stacked)
        print(f"✅ Saved 5-band GeoTIFF: {output_path}")




def batch_stack_5band(folder_in, folder_out  , post_append = '' ):
    os.makedirs(folder_out, exist_ok=True)

    r_files = sorted(glob.glob(os.path.join(folder_in, "*_R.png")))
    for r_path in r_files:
        base = os.path.basename(r_path).replace("_R.png", "")
        paths = {
            "r_path": r_path,
            "g_path": os.path.join(folder_in, base + "_G.png"),
            "b_path": os.path.join(folder_in, base + "_B.png"),
            "nir_path": os.path.join(folder_in, base + "_NIR.png"),
            "re_path": os.path.join(folder_in, base + "_RE.png"),
        }

        if not all(os.path.exists(p) for p in paths.values()):
            print(f"⚠️ Missing band(s) for {base}, skipping.")
            continue

        output_path = os.path.join(folder_out, base +  post_append + ".tif")
        try:
            stack_5band_geotiff(**paths, output_path=output_path)
        except Exception as e:
            print(f"❌ Failed to stack {base}: {e}")






def convert_uint16_to_uint8(img):
    """
    Converts a uint16 image (HWC) to uint8 using per-channel min-max scaling.
    """
    if img.ndim == 3:
        h, w, c = img.shape
        img_uint8 = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(c):
            band = img[:, :, i]
            b_min, b_max = band.min(), band.max()
            if b_max > b_min:
                scaled = 255 * (band - b_min) / (b_max - b_min)
            else:
                scaled = np.zeros_like(band)
            img_uint8[:, :, i] = scaled.astype(np.uint8)
        return img_uint8
    elif img.ndim == 2:
        b_min, b_max = img.min(), img.max()
        if b_max > b_min:
            scaled = 255 * (img - b_min) / (b_max - b_min)
        else:
            scaled = np.zeros_like(img)
        return scaled.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported shape: {img.shape}")


def save_uint8_tiff_auto_compression(image_uint8, output_path, photometric="minisblack"):
    """
    Save uint8 TIFF image using LZW if available, else no compression.
    """
    has_imagecodecs = find_spec("imagecodecs") is not None
    compression = 'LZW' if has_imagecodecs else None

    if not has_imagecodecs:
        print("⚠️ imagecodecs not found – saving without compression")

    tifffile.imwrite(
        output_path,
        image_uint8,
        dtype='uint8',
        photometric=photometric,
        planarconfig='CONTIG',
        compression=compression
    )

    print(f"✅ Saved: {output_path.name} | shape: {image_uint8.shape} | dtype: uint8 | compression: {compression or 'None'}")



def convert_folder_uint16_to_uint8(input_folder, output_folder, method='stretch'):
    """
    Convert all uint16 TIFF images in input_folder to uint8 and save in output_folder.
    Applies LZW compression if imagecodecs is available.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tiff_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.tiff"))
    print(f"📂 Found {len(tiff_files)} TIFF files in: {input_folder}")

    for tif_path in tiff_files:
        print(f"\n🔄 Processing: {tif_path.name}")
        try:
            image = tifffile.imread(tif_path)
            print(f"🔍 Loaded shape: {image.shape}, dtype: {image.dtype}")

            # Auto-detect (bands, H, W) format and transpose if needed
            if image.ndim == 3 and image.shape[0] < 10 and image.shape[1] > 50 and image.shape[2] > 50:
                image = np.transpose(image, (1, 2, 0))
                print("↪️ Reordered (bands, H, W) → (H, W, bands)")

            if image.ndim != 3 or image.dtype != np.uint16:
                print("⚠️ Skipped: Not a 3D uint16 image")
                continue

            num_channels = image.shape[2]
            print(f"📊 Detected {num_channels} channels")

            # Convert to uint8
            if method == 'stretch':
                image_uint8 = np.zeros_like(image, dtype=np.uint8)
                for i in range(num_channels):
                    band = image[..., i]
                    bmin, bmax = np.percentile(band, 1), np.percentile(band, 99)
                    scaled = np.clip((band - bmin) * 255 / (bmax - bmin + 1e-5), 0, 255)
                    image_uint8[..., i] = scaled.astype(np.uint8)
            elif method == 'normalize':
                image_uint8 = (image / 256).clip(0, 255).astype(np.uint8)
            else:
                raise ValueError("❌ Invalid method: choose 'stretch' or 'normalize'")

            # Set photometric type
            photometric = "rgb" if num_channels == 3 else "minisblack"
            out_path = output_folder / tif_path.name
            save_uint8_tiff_auto_compression(image_uint8, out_path, photometric)

        except Exception as e:
            print(f"❌ Error processing {tif_path.name}: {e}")





######################################################3  split train val  test       R ##############################################
def sort_images_from_split_files (txt_file_to_sort , img_folder ,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Open the file in read mode
    with open(txt_file_to_sort, 'r') as file:
        for line in file:
            # Strip newline characters and print the line
            current_file_line = line.strip()
            print(current_file_line)

            img_from = os.path.join(img_folder , current_file_line )

            # Loop through all files in the source directory   
            for filename in os.listdir(img_folder):
                if filename.startswith(current_file_line):
                    full_src_path = os.path.join(img_folder, filename)
                    full_dst_path = os.path.join(output_folder , filename)
                    shutil.copy2(full_src_path, full_dst_path)  # copy2 preserves meta=
                    print(f"Copied: {filename}")


                    







def merge_files(input_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for folder in input_folders:
        if not os.path.isdir(folder):
            print(f"❌ Skipping invalid folder: {folder}")
            continue
        for filename in os.listdir(folder):
            src = os.path.join(folder, filename)
            dst = os.path.join(output_folder, filename)

            # If file already exists in destination, rename it to avoid overwriting
            if os.path.exists(dst):
                name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{name}_{counter}{ext}"
                    dst = os.path.join(output_folder, new_name)
                    counter += 1

            shutil.copy2(src, dst)
            print(f"✅ Copied: {src} → {dst}")







def stack_RGB_set_uint8_sort_train_val_test (images_merge_band_images , uint16_to_uint8_method , labels_path_to_merge_RGB):

    #input_folder_RGB = "./weedsgalore-dataset/images_merge/"
    #output_folder_RGB_uint8 = "./weedsgalore-dataset/RGB/RGB_uint16/"

    split_files_path  = os.path.normpath(images_merge_band_images)
    # Return the parent directory
    split_files_path = os.path.dirname(split_files_path)
    split_files_path = os.path.join(split_files_path , 'splits' )

    # temp  stack uint16
    #temp_output_folder_RGB_uint16 =  "./weedsgalore-dataset/RGB/RGB_uint16/"
    temp_output_folder_RGB_uint16 =  labels_path_to_merge_RGB +  "/RGB_uint16/"
    
    # Stack RGB bands
    batch_stack_rgb_triplets(images_merge_band_images, temp_output_folder_RGB_uint16)

    #folder_RGB_uint8 = './weedsgalore-dataset/RGB/RGB_uint8_' + uint16_to_uint8_method + '/'
    folder_RGB_uint8 = labels_path_to_merge_RGB +  '//RGB_uint8_' + uint16_to_uint8_method + '/'
     
    convert_folder_uint16_to_uint8(temp_output_folder_RGB_uint16, folder_RGB_uint8, method=uint16_to_uint8_method)

    train_val_test_list = ['train' , 'val', 'test']
    #output_RGB_train_val_test_path = './weedsgalore-dataset/train_val_test/RGB/'
    output_RGB_train_val_test_path =  labels_path_to_merge_RGB         #'./weedsgalore-dataset/train_val_test/RGB/'
    for train_val_test in train_val_test_list: 

        #  RGB  ############################################################################################
        #txt_split_file_to_sort =  labels_path_to_merge_RGB +  './weedsgalore-dataset/weedsgalore-dataset/splits/' + train_val_test + '.txt'    # this folder have train/val/test split used for experimental traonong
        txt_split_file_to_sort = split_files_path + '/' + train_val_test + '.txt'    # this folder have train/val/test split used for experimental traonong
        #input_instances_folder =  folder_RGB_uint8   #'./weedsgalore-dataset/RGB/RGB_uint8_normalize/'
        output_folder_instances = output_RGB_train_val_test_path + train_val_test + '/images/'

        #sort_images (txt_file_to_sort ,input_semantics_folder ,  output_folder_semantics)
        sort_images_from_split_files (txt_split_file_to_sort ,folder_RGB_uint8 ,  output_folder_instances)
    return output_RGB_train_val_test_path

def stack_RGBRN_set_uint8_sort_train_val_test (images_merge_band_images , uint16_to_uint8_method , labels_path_to_merge_RGBRN ):
    
    split_files_path  = os.path.normpath(images_merge_band_images)
    # Return the parent directory
    split_files_path = os.path.dirname(split_files_path)
    split_files_path = os.path.join(split_files_path , 'splits' )

    # temp  stack uint16
    #temp_output_folder_RGB_uint16 =  "./weedsgalore-dataset/RGBRN/RGBRN_uint16/"
    temp_output_folder_RGBRN_uint16 =  labels_path_to_merge_RGBRN +  "/RGBRN_uint16/"
    
    # Stack RGB bands
    batch_stack_5band (images_merge_band_images, temp_output_folder_RGBRN_uint16)

    #folder_RGB_uint8 = './weedsgalore-dataset/RGBRN/RGBRN_uint8_' + uint16_to_uint8_method + '/'
    folder_RGB_uint8 = labels_path_to_merge_RGBRN +  '//RGBRN_uint8_' + uint16_to_uint8_method + '/'
     
    convert_folder_uint16_to_uint8(temp_output_folder_RGBRN_uint16, folder_RGB_uint8, method=uint16_to_uint8_method)

    train_val_test_list = ['train' , 'val', 'test']
    #output_RGB_train_val_test_path = './weedsgalore-dataset/train_val_test/RGB/'
    output_RGBRN_train_val_test_path =  labels_path_to_merge_RGBRN         #'./weedsgalore-dataset/train_val_test/RGB/'
    for train_val_test in train_val_test_list: 

        #  RGB  ############################################################################################
        #txt_split_file_to_sort =  labels_path_to_merge_RGBRN +  './weedsgalore-dataset/weedsgalore-dataset/splits/' + train_val_test + '.txt'    # this folder have train/val/test split used for experimental traonong
        txt_split_file_to_sort = split_files_path + '/' + train_val_test + '.txt'    # this folder have train/val/test split used for experimental traonong
        #input_instances_folder =  folder_RGB_uint8   #'./weedsgalore-dataset/RGB/RGB_uint8_normalize/'
        output_folder_instances = output_RGBRN_train_val_test_path + train_val_test + '/images/'

        #sort_images (txt_file_to_sort ,input_semantics_folder ,  output_folder_semantics)
        sort_images_from_split_files (txt_split_file_to_sort ,folder_RGB_uint8 ,  output_folder_instances)
    return output_RGBRN_train_val_test_path







    # Step 3.  Stack RGBRN images and convert from RGBRN uint16 to uint8   #######################################################3
    #input_folder_RGBRN = "./weedsgalore-dataset/images_merge/"
    temp_output_folder_RGBRN_uint16 = "./weedsgalore-dataset/RGBRN/RGBRN_uint16/"

    # Stack RGBNR bands
    batch_stack_5band(images_merge_band_images, temp_output_folder_RGBRN_uint16)

    folder_RGBRN_uint8 = './weedsgalore-dataset/RGBRN/RGBRN_uint8_' + uint16_to_uint8_method + '/'
    convert_folder_uint16_to_uint8(temp_output_folder_RGBRN_uint16, folder_RGBRN_uint8, method=uint16_to_uint8_method)

    train_val_test_list = ['train' , 'val', 'test']
    output_RGBRN_train_val_test_path = './weedsgalore-dataset/train_val_test/RGBRN/'
    for train_val_test in train_val_test_list: 
        #  RGBRN  ############################################################################################
        txt_split_file_to_sort = './weedsgalore-dataset/weedsgalore-dataset/splits/' + train_val_test + '.txt'
        #input_instances_folder = './weedsgalore-dataset/RGBRN/RGBRN_uint8_normalize/'
        output_folder_instances = output_RGBRN_train_val_test_path  + train_val_test + '/images/'

        #sort_images (txt_file_to_sort ,input_semantics_folder ,  output_folder_semantics)
        sort_images_from_split_files (txt_split_file_to_sort ,folder_RGBRN_uint8 ,  output_folder_instances)

    return output_RGBRN_train_val_test_path








# def main():
#     create_RGB = True
#     if create_RGB:
#         # === Example Usage ===    RGB
#         input_folder = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/images_merge/"
#         output_folder_uint16 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGB/RGB_uint16/"
#         batch_stack_rgb_triplets(input_folder, output_folder_uint16)

#         output_folder_uint8 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGB/RGB_uint8_stretch/"
#         convert_folder_uint16_to_uint8(output_folder_uint16, output_folder_uint8, method='stretch')

#         output_folder_uint8 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGB/RGB_uint8_normalize/"
#         convert_folder_uint16_to_uint8(output_folder_uint16, output_folder_uint8, method='normalize')

#     create_RGBNR = False
#     if create_RGBNR:
#         # === Example Usage ===    RGB/NIR/RE
#         input_folder = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/images_merge/"
#         output_folder_uint16 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGBRN/RGBRN_uint16/"
#         batch_stack_5band(input_folder, output_folder_uint16)

#         output_folder_uint8 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGBRN/RGBRN_uint8_stretch/"
#         convert_folder_uint16_to_uint8(output_folder_uint16, output_folder_uint8, method='stretch')

#         output_folder_uint8 = "D:/PD/Publications/Yolo_mod/github/YOLO-Multispectral/examples/notebooks/weedsgalore-dataset/RGBRN/RGBRN_uint8_normalize/"
#         convert_folder_uint16_to_uint8(output_folder_uint16, output_folder_uint8, method='normalize')





# if __name__ == '__main__':
#     main()

