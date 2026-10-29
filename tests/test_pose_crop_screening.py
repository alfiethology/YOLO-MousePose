"""Regression checks for conservative focal-rat crop screening."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/dataset'))
from prepare_rat_crops import crop_bounds, focal_match, neighbour_hits, KEYPOINTS, FLIP_IDX, yolo_row


class CropScreeningTests(unittest.TestCase):
    def test_neighbour_must_enter_actual_crop(self):
        proposals=[{'box':[20,20,80,80]}, {'box':[100,0,120,80]}, {'box':[99,0,120,80]}]
        hits=neighbour_hits([0,0,100,100],proposals,0,16)
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['overlap_pixels'],80)

    def test_only_matched_focal_box_is_exempt(self):
        proposals=[{'box':[20,20,60,60]},{'box':[0,0,100,100]}]
        match=focal_match([0,0,100,100],proposals,0.3)
        self.assertEqual(match,1)
        self.assertEqual(len(neighbour_hits([0,0,100,100],proposals,match,16)),1)
        self.assertIsNone(focal_match([200,200,300,300],proposals,0.3))

    def test_crop_preserves_labelled_landmarks_and_masks_absent_points(self):
        shape={'shape_type':'skeleton','points':[[10,10],[20,10],[20,20],[10,20],
                   [8,15],[15,15],[15,15],[15,15],[15,15],[999,999]],
               'pose':{'keypoints':KEYPOINTS,'flip_idx':FLIP_IDX,'visibility':[1,2,2,2,2,0]}}
        bounds=crop_bounds(shape,100,100,2)
        self.assertEqual(bounds,[6,8,22,22])
        shifted=dict(shape,points=[[x-bounds[0],y-bounds[1]] for x,y in shape['points']])
        fields=yolo_row(shifted,16,14).split()
        self.assertEqual(len(fields),23)
        self.assertEqual([float(v) for v in fields[-3:]],[0.,0.,0.])
        self.assertEqual(float(fields[7]),1.)


if __name__=='__main__':unittest.main()
