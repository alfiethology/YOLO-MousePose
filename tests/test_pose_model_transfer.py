"""Fast CPU checks for pose head layout, missing landmarks and safe transfer."""

import copy
import unittest

import torch
from torch import nn

from models.yolo import Detect
from utils.loss import ComputeLoss
from utils.pose_transfer import MOUSE_KEYPOINTS, transfer_pose_weights


RAT = ['snout', 'left_ear', 'right_ear', 'neck_base', 'body_centre', 'tail_base']


class SmallPose(nn.Module):
    def __init__(self, names=RAT, layout='per_anchor'):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(4, 4, 1),
                                   Detect(1, [[10, 13, 16, 30, 33, 23]], len(names), [4])])
        self.model[-1].stride = torch.tensor([8.])
        self.yaml = {'kpt_names': list(names), 'pose_anchor_layout': layout}
        self.gr = 1.
        self.hyp = dict(cls_pw=1., obj_pw=1., fl_gamma=0., anchor_t=4.,
                        box=0.05, obj=1., cls=0.5, kpt=0.1)


class PoseModelTests(unittest.TestCase):
    def test_each_anchor_contains_its_own_detection_and_landmark_channels(self):
        head = SmallPose().model[-1]
        with torch.no_grad():
            head.m[0].weight.zero_()
            head.m_kpt[0].weight.zero_()
            head.m[0].bias.copy_(torch.arange(18.))
            head.m_kpt[0].bias.copy_(torch.arange(54.)+100)
        head.train()
        raw = head([torch.zeros(1, 4, 2, 2)])[0]
        for anchor in range(3):
            torch.testing.assert_close(raw[0, anchor, 0, 0, :6], torch.arange(6.)+6*anchor)
            torch.testing.assert_close(raw[0, anchor, 0, 0, 6:], torch.arange(18.)+18*anchor+100)

    def test_inference_landmark_confidences_are_probabilities(self):
        head = SmallPose().model[-1]
        with torch.no_grad():
            head.m_kpt[0].weight.zero_()
            head.m_kpt[0].bias.fill_(-2.)
        head.eval()
        predictions, _ = head([torch.zeros(1, 4, 2, 2)])
        self.assertEqual(predictions.shape, (1, 12, 24))
        torch.testing.assert_close(predictions[..., 8::3], torch.full((1, 12, 6), torch.sigmoid(torch.tensor(-2.))))

    def test_absent_landmarks_have_no_coordinate_gradient(self):
        model = SmallPose()
        loss_fn = ComputeLoss(model, kpt_label=True)
        predictions = [torch.zeros(1, 3, 8, 8, 24, requires_grad=True)]
        targets = torch.tensor([[0., 0., .5, .5, .5, .5]+[0.]*12])
        loss, parts = loss_fn(predictions, targets, True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(parts[3]), 0.)
        loss.backward()
        gradients = predictions[0].grad[..., 6:].reshape(1, 3, 8, 8, 6, 3)
        self.assertEqual(float(gradients[..., :2].abs().sum()), 0.)
        self.assertGreater(float(gradients[..., 2].abs().sum()), 0.)

    def test_annotated_point_on_image_edge_is_not_absent(self):
        model = SmallPose()
        loss_fn = ComputeLoss(model, kpt_label=True)
        predictions = [torch.zeros(1, 3, 8, 8, 24, requires_grad=True)]
        targets = torch.tensor([[0., 0., .5, .5, .5, .5, 0., .3]+[0.]*10])
        _, _, _, _, _, masks = loss_fn.build_targets(predictions, targets)
        self.assertTrue(masks[0][:, 0].any())
        self.assertFalse(masks[0][:, 1:].any())
        loss, _ = loss_fn(predictions, targets, True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_shared_landmark_channels_remap_and_neck_stays_initialized(self):
        source, target = SmallPose(MOUSE_KEYPOINTS), SmallPose()
        source_head, target_head = source.model[-1], target.model[-1]
        with torch.no_grad():
            source_head.m_kpt[0].bias.copy_(torch.arange(63.))
        original = target_head.m_kpt[0].bias.detach().clone()
        report = transfer_pose_weights(target, source)
        self.assertEqual(report['new_landmarks'], ['neck_base'])
        self.assertEqual(len(report['copied_landmarks']), 5)
        current = target_head.m_kpt[0].bias
        for anchor in range(3):
            for new, name in enumerate(RAT):
                start = (anchor*6+new)*3
                if name == 'neck_base':
                    torch.testing.assert_close(current[start:start+3], original[start:start+3])
                else:
                    old = (anchor*7+list(MOUSE_KEYPOINTS).index(name))*3
                    torch.testing.assert_close(current[start:start+3], source_head.m_kpt[0].bias[old:old+3])

    def test_legacy_anchor_layout_transfers_backbone_but_not_pose_head(self):
        source, target = SmallPose(MOUSE_KEYPOINTS, 'legacy'), SmallPose()
        original = copy.deepcopy(target.model[-1].state_dict())
        report = transfer_pose_weights(target, source)
        self.assertTrue(report['legacy_pose_head_skipped'])
        self.assertTrue(report['loaded_tensors'])
        for key, value in target.model[-1].state_dict().items():
            torch.testing.assert_close(value, original[key])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
