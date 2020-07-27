from rtree import index
import shapely

class rtree_utils():
    def __init__(self, saved_path='', bounds = [], override=True):
        p = index.Property()
        p.overwrite = override
        self.r_idx = index.Index(saved_path, properties=p)

        for bound in bounds:
            ID, left, bottom, right, top = bound
            # print( ID, left, bottom, right, top)
            self.r_idx.insert(ID, (left, bottom, right, top))

        # index.Rtree(saved_path)   # save the rtree.
        # self.r_idx.close()


    # def build_RtreeIdx(self, bounds, saved_name):  # saved_name has no suffix
    #     """
    #     :param bounds: a list of (id, left, bottom, right, top)
    #     :param saved_name: saved_name has no suffix
    #     :return: R-tree
    #     """
    #     r_idx = index.Index(saved_name)
    #     for bound in bounds:
    #         ID, left, bottom, right, top = bound
    #         # print( ID, left, bottom, right, top)
    #         r_idx.insert(ID, (left, bottom, right, top))
    #     r_idx.close()
    #     return r_idx

    # def load_RtreeIdx(self, saved_name):  # saved_name has no suffix
    #     """
    #     :param saved_name: saved_name has no suffix
    #     :return: R-tree
    #     """
    #     r_idx = index.Index(saved_name)
    #     return r_idx

    def isInBounds(self, bounds, Rtree_idx):
        """

        :param bounds: (left, bottom, right, top)
        :param Rtree_idx:
        :return:
        """
        intersects = len(list(Rtree_idx.intersection(bounds)))
        if intersects > 0:
            return True
        else:
            return False
    #
    # def isIntersect(self, polygon: shapely.geometry.Polygon) -> bool:
    # # polyon: Shapely Polygon
    #     bound = polygon.bouds   #Returns a (minx, miny, maxx, maxy)
    #     results = self.r_idx.intersection(bound)
    #     results = list(results)
    #     if results > 0:
    #         return False



def test_init():
    bounds = [[0, -1, 0, 1, 2]] # ID should be integate, ID, left, bottom, right, top (bottom_left, upper_right)
    # rtree_u = rtree_utils(saved_path=r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\rtree3', bounds=bounds, override=False)
    rtree_u = rtree_utils(saved_path=r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\rtree3', override=False)
    re = rtree_u.r_idx.intersection((-0.5, -1, 0.5, 1))
    print(list(re))

if __name__ == "__main__":
    test_init()

