import os
import pygrib
import h5py
import numpy as np
import pickle

# Get a list of all files in the folder
def TransGrib2H5(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.grib2')]
    output_file_paths = []

    for hour_value in range(24):
        hour_str = f't{hour_value:02d}z'
        for file_name in file_list:
            parts = file_name.split('.')

            if hour_str in parts:
                # Check if corresponding .h5 file already exists
                output_file_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.h5')
                if os.path.exists(output_file_path):
                    output_file_paths.append(output_file_path)
                    print(f"Skipping {file_name}, .h5 file already exists.")
                    continue

                # Open the GRIB file
                file_path = os.path.join(folder_path, file_name)
                grbs = pygrib.open(file_path)

                # Select and process the desired messages
                grb_list = grbs.select(name='Total Precipitation', typeOfLevel='surface')
                if grb_list:
                    for grb in grb_list:
                        # Process the GRIB message as needed
                        print(f"File: {file_name}")
                        print("Values Shape:", grb.values.shape)
                        data = grb.values
                        lats, lons = grb.latlons()

                        with h5py.File(output_file_path, 'w') as h5_file:
                            h5_file.create_dataset('fields', data=data)
                            h5_file.create_dataset('lats', data=lats)
                            h5_file.create_dataset('lons', data=lons)

                        print(f'save to: {output_file_path}')
                        output_file_paths.append(output_file_path)
                else:
                    print(f"No matching messages found for {file_name}")

                # Close the GRIB file
                grbs.close()

    return output_file_paths

def CropH5(file_paths):
    result_file_path = 'result.h5'
    with h5py.File(result_file_path, 'w') as result_file:
        data_index = 0  # 记录数据的索引
        for i, file_path in enumerate(file_paths):
            # 检查文件路径是否包含 'f00'
            if 'f00' in file_path:
                continue
            with h5py.File(file_path, 'r') as file:
                fields_dataset = file['fields']
                fields_data = fields_dataset[:]

            start_row = (fields_data.shape[0] - 1052) // 2
            start_col = (fields_data.shape[1] - 1788) // 2
            end_row = start_row + 1052
            end_col = start_col + 1788

            # 将fields_data进行中心裁剪
            fields_data = fields_data[start_row:end_row, start_col:end_col]
            result_file.create_dataset(f'data_{data_index}', data=fields_data)
            data_index += 1 

    with h5py.File(result_file_path, 'r') as result_file:
        result_data = np.array([result_file[f'data_{i}'][:] for i in range(data_index)])


    print(result_data.shape) 
    print(result_data.dtype)  
    print(result_data[:3])  

if __name__ == '__main__':
    # Specify the folder containing the GRIB files
    folder_path = 'C:/Users/39349/MyHRRR' ## Grib2 文件存放路径
    file_paths = TransGrib2H5(folder_path)
    CropH5(file_paths)

