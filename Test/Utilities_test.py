import unittest
import Utilities
import pandas as pd
import glob
import os
import shutil
import time

class MyTestCase(unittest.TestCase):
    download_files = Utilities.Download_files()

    file1 = r'G:\My Drive\Sidewalk_extraction\Essex\panos\IJfUXDE9hSi1lX5eTLRRmA.jpg'
    file2 = r'G:\My Drive\Sidewalk_extraction\Essex\panos\JSAarKXRO_ecO3sJIFTkFw.jpg'
    saved_path = r'O:\Essex'
    file_list = [file1, file2]
    def test_download_from_list(self, file_list=file_list, saved_path=saved_path, report_cnt = 4):
        print(file_list)
        self.download_files.download_from_list(file_list, saved_path)
        new_file_names = [os.path.join(saved_path, os.path.basename(file)) for file in file_list]
        print(new_file_names)
        for file in new_file_names:
            print(file)
            self.assertEqual(True, os.path.exists(file))
            os.remove(file)
            print(file)




if __name__ == '__main__':
    unittest.main()
