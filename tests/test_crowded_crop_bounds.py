import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/dataset'))
from prepare_rat_crowded_crops import whole_neighbour_bounds,local_target

def rect(box):
    a,b,c,d=box
    return {'shape_type':'rectangle','points':[[a,b],[c,d]]}

def pose(box):
    a,b,c,d=box
    return {'shape_type':'skeleton','points':[[a,b],[c,b],[c,d],[a,d]]+[[a+1,b+1]]*6,
            'pose':{'visibility':[2]*6}}

class CrowdedCropBoundsTests(unittest.TestCase):
    def test_whole_neighbours_expand_transitively(self):
        focal=pose([10,10,40,40]);near=rect([42,20,80,50]);next_rat=rect([82,30,110,60]);far=rect([170,170,190,190])
        bounds=whole_neighbour_bounds(focal,[next_rat,near,focal,far],200,200,5)
        assert bounds==[5,5,115,65]
        assert local_target(near,bounds)['points'][:4]==[[37,15],[75,15],[75,45],[37,45]]

    def test_single_rat_keeps_padding_and_image_limits(self):
        focal=pose([1,1,20,20])
        assert whole_neighbour_bounds(focal,[focal],100,100,10)==[0,0,30,30]
