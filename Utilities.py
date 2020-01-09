import pandas as pd
import glob
import os
import shutil
import time
import multiprocessing as mp

class Download_files():
    def download_from_list(self, list_files, saved_path, report_cnt=10):
        pro_id = os.getpid()
        start_time = time.time()
        total_cnt = len(list_files)
        processed_cnt =  0
        while len(list_files) > 0:
            try:
                file = list_files.pop(0)
                print("Processing:", file)
                processed_cnt += 1
                basename = os.path.basename(file)
                new_file_name = os.path.join(saved_path, basename)
                shutil.copyfile(file, new_file_name)

                # print(new_file_name)

                if processed_cnt % report_cnt == 0:
                    used_time = time.time() - start_time
                    cnt_hour = int(processed_cnt / (used_time + 1) * 3600)
                    print(f"Processing speed: {processed_cnt} / {total_cnt}, Processing speed: {cnt_hour} item / hour,  pid: {pro_id}")
            except Exception as e:
                print("Error in download_from_list: ", file, e)
                continue

    def download_from_list_mp(self, list_files, saved_path, report_cnt=10, Process_cnt=5):
        list_files_mp = mp.Manager().list()
        for i in list_files:
            list_files_mp.append(i)
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(self.download_from_list, args=(list_files_mp, saved_path, report_cnt))
        pool.close()
        pool.join()

if __name__ == "__main__":

    f = open(r'D:\Essex\jsons_from_panos\json_list.txt', 'r')
    saved_path = r'D:\Essex\t'
    list_files = ['G:\\My Drive\\Sidewalk_extraction\Essex\\panos\\' + line.replace('.json\n', '.json') for line in f]
    print("len of list:", len(list_files))
    download = Download_files()
    download.download_from_list_mp(list_files,saved_path)
    # for cnt, line in enumerate(f):

