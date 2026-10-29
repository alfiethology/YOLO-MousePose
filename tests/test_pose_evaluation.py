"""Regression checks for multi-animal rat pose evaluation and inference."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test import match_instances, test as validate
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_skeleton_kpts, plot_results


class KnownPose(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        head = torch.nn.Identity()
        head.nkpt = 6
        self.model = torch.nn.ModuleList([head])
        self.stride = torch.tensor([32.])
        self.names = ['rat']
        self.yaml = {'nc': 1, 'nkpt': 6, 'flip_idx': [0, 2, 1, 3, 4, 5]}
        self.predictions = predictions

    def forward(self, images, augment=False):
        return self.predictions.to(images), []


class PoseEvaluationTests(unittest.TestCase):
    def test_nms_keeps_each_animals_keypoints_and_confidence(self):
        prediction = torch.zeros((1, 3, 24))
        prediction[0, :, :4] = torch.tensor([[20, 20, 20, 20], [80, 80, 20, 20], [20, 20, 20, 20]])
        prediction[0, :, 4:6] = torch.tensor([[.9, .8], [.95, .9], [.6, .7]])
        for index in range(3):
            prediction[0, index, 6:] = torch.arange(18) + 100*index
        output = non_max_suppression(prediction, .1, kpt_label=True, nc=1, nkpt=6)[0]
        self.assertEqual(tuple(output.shape), (2, 24))
        self.assertTrue(torch.allclose(output[:, 4], torch.tensor([.855, .72])))
        self.assertTrue(torch.equal(output[0, 6:], torch.arange(18)+100))
        matches = match_instances(output, torch.zeros(2), torch.tensor([[10., 10, 30, 30], [70, 70, 90, 90]]))
        self.assertEqual({(p, t) for p, t, _ in matches}, {(0, 1), (1, 0)})

    def test_missing_points_survive_rescaling_and_sixpoint_drawing(self):
        coords = torch.tensor([[0., 0., 20., 30., 40., 50.]])
        scale_coords((640, 640), coords, (972, 1296), ((.493827, .493827), (0., 80.)), kpt_label=True)
        self.assertTrue(torch.equal(coords[0, :2], torch.zeros(2)))
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        plot_skeleton_kpts(image, [20, 20, 15, 30, 25, 30, 20, 40, 20, 60, 20, 80], 2)
        self.assertGreater(image.sum(), 0)

    def test_multi_animal_pck_matches_instances_and_counts_misses(self):
        # Predicted animal order is reversed: its points must follow geometric matching.
        targets = torch.tensor([[0., 0., .25, .25, .375, .375, *([.25, .25]*6)],
                                [0., 0., .75, .75, .375, .375, *([.75, .75]*6)]])
        predictions = torch.tensor([[[24., 24., 12., 12., .99, .99, *([24., 24., .99]*6)],
                                     [8., 8., 12., 12., .95, .99, *([8., 8., .99]*6)]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = dict(train=str(path), val=str(path), nc=1, names=['rat'], nkpt=6,
                        kpt_names=['snout', 'left_ear', 'right_ear', 'neck_base', 'body_centre', 'tail_base'])
            loader = [(torch.zeros((1, 3, 32, 32), dtype=torch.uint8), targets,
                       ['frame.jpg'], [((32, 32), ((1., 1.), (0., 0.)))])]
            results, _, _ = validate(data, model=KnownPose(predictions), dataloader=loader,
                                     save_dir=path, plots=False, kpt_label=True, half_precision=False, save_txt=True)
            lines = (path/'labels/frame.txt').read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[0].split()), 23)  # cls+bbox+six triplets
            self.assertAlmostEqual(float(lines[0].split()[5]), .75)
            self.assertEqual(len(results), 28)
            report = json.loads((path/'pose_metrics.json').read_text())
            self.assertEqual(report['matched_keypoint_error_px'], 0)
            self.assertEqual(report['pck_by_bbox_diagonal']['0.10'], 1)
            validate(data, model=KnownPose(predictions[:, :1]), dataloader=loader,
                     save_dir=path, plots=False, kpt_label=True, half_precision=False)
            report = json.loads((path/'pose_metrics.json').read_text())
            self.assertEqual(report['pck_by_bbox_diagonal']['0.10'], .5)
            self.assertEqual(report['matched_points'], [1.]*6)
            self.assertEqual(report['labelled_points'], [2.]*6)

    def test_named_metric_csv_plots_without_sevenpoint_assumptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'metrics.csv').write_text('epoch,train/box_loss,val/pck_diag_0.10,val/matched_keypoint_error_px\n1,1.2,0.6,8.0\n2,1.0,0.7,7.0\n')
            plot_results(save_dir=path)
            self.assertTrue((path/'results.png').is_file())


if __name__ == '__main__':
    unittest.main()
