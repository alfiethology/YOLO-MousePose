"""Prevent visible animals losing targets during tight-crop augmentation."""

import unittest
from unittest.mock import patch

import numpy as np

from utils.datasets import box_candidates, random_perspective


class PoseAugmentationTests(unittest.TestCase):
    def test_clipped_visible_animal_remains_a_target(self):
        source=np.array([[10.],[10.],[90.],[90.]])
        transformed=np.array([[-20.],[-20.],[120.],[120.]])
        self.assertTrue(box_candidates(source,transformed,100,100,ol_thr=.01)[0])
        self.assertFalse(box_candidates(source,transformed,100,100,ol_thr=.8)[0])

    def test_fully_outside_animal_is_removed(self):
        source=np.array([[10.],[10.],[90.],[90.]])
        transformed=np.array([[110.],[10.],[190.],[90.]])
        self.assertFalse(box_candidates(source,transformed,100,100,ol_thr=.01)[0])

    def test_pose_zoom_keeps_box_and_masks_only_outside_landmarks(self):
        image=np.zeros((100,100,3),dtype=np.uint8)
        targets=np.array([[0,5,5,95,95, 10,50, 40,40, 60,40, 50,50, 50,60, 90,50]],dtype=np.float32)
        # Deterministically zoom by 1.5, with all other affine terms zero.
        sequence=[0.,0.,0.,1.5,0.,0.,.5,.5]
        with patch('utils.datasets.random.uniform',side_effect=sequence):
            _,result=random_perspective(image,targets,degrees=0,translate=0,scale=.5,shear=0,perspective=0,kpt_label=True)
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0,1:5],[0,0,100,100])
        np.testing.assert_allclose(result[0,5:7],[0,0])
        np.testing.assert_allclose(result[0,7:9],[35,35])
        np.testing.assert_allclose(result[0,-2:],[0,0])


if __name__=='__main__':unittest.main()
