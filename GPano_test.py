import unittest
import GPano


class MyTestCase(unittest.TestCase):
    def test_castesian_to_shperical(self):
        self.assertEqual(True, GPano.GPano.castesian_to_shperical(self, (383, 512), 768, 1024, 150, 0, 90))


if __name__ == '__main__':
    unittest.main()
