"""MousePose must not read or overwrite the shared Ultralytics label cache."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from utils.datasets import LoadImagesAndLabels

class PoseCacheTests(unittest.TestCase):
    def test_ultralytics_cache_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);images=root/'images/train';labels=root/'labels/train'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            Image.new('RGB',(64,64)).save(images/'rat.jpg')
            (labels/'rat.txt').write_text('0 .5 .5 .8 .8 '+'.4 .4 2 '*6+'\n')
            shared=labels.with_suffix('.cache');shared.write_bytes(b'other trainers cache format')
            dataset=LoadImagesAndLabels(str(images),img_size=64,batch_size=1,nkpt=6,kpt_label=True)
            self.assertEqual(len(dataset),1)
            self.assertEqual(shared.read_bytes(),b'other trainers cache format')
            self.assertTrue(labels.with_suffix('.mousepose.cache').is_file())
