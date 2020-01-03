import pandas as pd
import glob
import os
import shutil
import time

class Download_files():
    def download_from_list(self, list_files, saved_path, report_cnt=10):
        pro_id = os.getpid()
        start_time = time.time()
        total_cnt = len(list_files)
        processed_cnt =  0
        while len(list_files) > 0:
            file = list_files.pop(0)
            print("Processing:", file)
            processed_cnt += 1
            basename = os.path.basename(file)
            new_file_name = os.path.join(saved_path, basename)
            shutil.copyfile(file, new_file_name)

            if processed_cnt % report_cnt == 0:
                used_time = time.time() - start_time
                cnt_hour = int(processed_cnt / (used_time + 0.000001) * 3600)
                print(f"Processed: {file}, {processed_cnt} / {total_cnt}, Processing speed: {cnt_hour} item / hour,  pid: {pro_id}")

