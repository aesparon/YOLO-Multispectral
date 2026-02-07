import os
import pandas as pd
from pathlib import Path
import re


def get_last_value(csv_path, column_name='Box_mAP@0.5'):
    try:
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            last_value = df[column_name].dropna().iloc[-1]
            return last_value
        else:
            raise KeyError(f"Column '{column_name}' not found in {csv_path}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None
    

def summary_results(process_folder):
    root_dir = Path(process_folder)



    summary = []

    for folder in root_dir.iterdir():
        if folder.is_dir():
            results_path = str(folder / "results.csv")
            if  os.path.exists(results_path):                                                   #   results_path.exists():
                try:
                    df = pd.read_csv(results_path)

                    best_idx = df["metrics/mAP50(B)"].idxmax()
                    best_row = df.loc[best_idx]
                    last_row = df.iloc[-1]

                    match = re.search(r'yolo[^\\/]*-seg', folder.name)
                    model_name = match.group(0) if match else "unknown"

                    match_TL = re.search(r'dropblock_[^\\/]*_seg', folder.name)
                    match_TL_name = match_TL.group(0) if match_TL else "unknown"

                    # Get Box_mAP@0.5 from test CSVs
                    folder_base = folder.name

                    # Load test prediction results
                    out_test_last_csv = os.path.join(folder,'test_last', 'out.csv')
                    map50_all_last = get_last_value(out_test_last_csv, column_name='Box_mAP@0.5')
                    out_test_best_csv = os.path.join(folder,'test_best', 'out.csv')
                    map50_all_best = get_last_value(out_test_best_csv, column_name='Box_mAP@0.5')

                    #test_last_row = test_last_df[test_last_df['folder'].str.contains(folder_base)]
                    #test_best_row = test_best_df[test_best_df['folder'].str.contains(folder_base)]

                    #box_map_05_last = test_last_row['Box_mAP@0.5'].values[0] if not test_last_row.empty else None
                    #box_map_05_best = test_best_row['Box_mAP@0.5'].values[0] if not test_best_row.empty else None

                    summary.append({
                        "folder": folder.name,
                        "MAP50_Best": best_row["metrics/mAP50(B)"],
                        "MAP50_95_Best": best_row["metrics/mAP50-95(B)"],
                        "MAP50_Last": last_row["metrics/mAP50(B)"],
                        "MAP50_95_Last": last_row["metrics/mAP50-95(B)"],
                        "epochs": len(df),
                        "last_epoch": last_row["epoch"],
                        "model": model_name,
                        "TL": match_TL_name,
                        "Box_mAP@0.5_Best": map50_all_best,
                        "Box_mAP@0.5_Last": map50_all_last,
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {results_path}: {e}")

    summary_df = pd.DataFrame(summary)
    return summary_df





# # # Example usage
# # 'RGB_tests_final_seed42\v11'

# #base_folder_RGB = 'd:/PD/Publications/Yolo_mod/github/colab_test/notebooks/YOLO-Multispectral/datasets/weeds_galore_processed/RGBRN/outputs/_RGBRN_yolo11x-seg_pt_method_add_ch_avg_TL_seg'

# base_folder_RGB = 'd:/PD/Publications/Yolo_mod/github/YOLO-Multispectral_2025_12_12/examples/notebooks/YOLO-Multispectral/datasets/weeds_galore_processed/RGBRN/outputs/'
                
# #base_folder_RGB = 'd:/PD/Publications/Yolo_mod/github/YOLO-Multispectral_2025_12_12/examples/notebooks/YOLO-Multispectral/datasets/weeds_galore_processed/RGB//outputs/'

# result_RGB = summary_results(base_folder_RGB)
# print(result_RGB)


# aa=22








# #model='v8' 
# base_folder_RGB ="D:/PD/yolo_mod/working2/projects/weeds_galore/RGB_tests_final_seed42/v8/"
# result_RGB = summary_results(base_folder_RGB)
# result_RGB.to_csv(base_folder_RGB + 'results_RGB_v8.csv')

# base_folder_RGBRN ="D:/PD/yolo_mod/working2/projects/weeds_galore/RGBRN_tests_final_seed42/v8/"
# result_RGBRN = summary_results(base_folder_RGBRN)
# result_RGBRN.to_csv(base_folder_RGBRN + 'results_RGB_v8.csv')



# #model='v11' 
# base_folder_RGB ="D:/PD/yolo_mod/working2/projects/weeds_galore/RGB_tests_final_seed42/v11/"
# result_RGB = summary_results(base_folder_RGB)
# result_RGB.to_csv(base_folder_RGB + 'results_RGB_v11.csv')

# base_folder_RGBRN ="D:/PD/yolo_mod/working2/projects/weeds_galore/RGBRN_tests_final_seed42/v11/"
# result_RGBRN = summary_results(base_folder_RGBRN)
# result_RGBRN.to_csv(base_folder_RGBRN + 'results_RGB_v11.csv')
  


# # out_test_best_csv =  os.path.join(process_folder,'test_best', 'out.csv')
# # out_test_last_csv =  os.path.join(process_folder,'test_last', 'out.csv')
# #result = summary_results(process_folder, out_test_last_csv, out_test_best_csv)




# aa=22


# base_folder ="D:/PD/yolo_mod/working2/projects/weeds_galore/"



# model='v8'   
# process_folder = base_folder + "/RGB_tests_final_seed42/" + model + '/'
# result = summary_results(process_folder )





# model='v11'   
# process_folder = base_folder + "/RGB_tests_final_seed42/" + model + '/'
# df_add = summary_results(process_folder )
# result = pd.concat([result, df_add], ignore_index=True)





# model='v8'   
# process_folder = base_folder + "/RGBRN_tests_final_seed42/" + model + '/'
# df_add = summary_results(process_folder )
# result = pd.concat([result, df_add], ignore_index=True)

# model='v11'   
# process_folder = base_folder + "/RGBRN_tests_final_seed42/" + model + '/'
# df_add = summary_results(process_folder )
# result = pd.concat([result, df_add], ignore_index=True)



# result.to_csv(base_folder + 'compile_results2.csv')








# # 👁️ Display the summary
# #import ace_tools as tools; tools.display_dataframe_to_user(name="YOLO Training Summary", dataframe=summary_df)
