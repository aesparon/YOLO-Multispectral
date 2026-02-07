import rasterio

path = "2023-06-15_0735.tif"  # adjust path

with rasterio.open(path) as ds:
    print("Band count:", ds.count)
    for i in range(1, ds.count + 1):
        desc = ds.descriptions[i - 1]
        print(f"Band {i}: {desc}")


aa=22