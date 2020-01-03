import unittest
import Utilities
import pandas as pd
import glob
import os
import shutil
import time

class MyTestCase(unittest.TestCase):
    util = Utilities()
    def test_something(self):
        self.assertEqual(True, False)
    file1 = r'G:\My Drive\Sidewalk_extraction\Essex\panos\IJfUXDE9hSi1lX5eTLRRmA.jpg'
    file2 = r'G:\My Drive\Sidewalk_extraction\Essex\panos\JSAarKXRO_ecO3sJIFTkFw.jpg'
    file_list = ['file_list']
    saved_path = r'J:\temp'
    def test_download_from_list(self, file_list=file_list, saved_path, report_cnt = 10):
        util.download_from_list("")
        new_file_name
        self.assertEqual(True, os.path.exists(new_file_name))



if __name__ == '__main__':
    unittest.main()
