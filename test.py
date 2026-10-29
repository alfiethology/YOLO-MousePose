"""Detection and pose validation with one-to-one multi-animal matching.

BBox AP and keypoint PCK are reported separately. Keypoint metrics use labelled
points only; missed target animals count as incorrect for PCK, not as omissions.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (check_dataset, check_file, check_img_size, check_requirements,
                           box_iou, non_max_suppression, scale_coords, xyxy2xywh,
                           xywh2xyxy, set_logging, increment_path, colorstr)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


def match_instances(predictions, target_classes, target_boxes, threshold=0.5):
    """Return unique (prediction, target, IoU) pairs, strongest overlap first."""
    if not len(predictions) or not len(target_boxes):
        return []
    overlap = box_iou(predictions[:, :4], target_boxes)
    valid = (overlap >= threshold) & (predictions[:, 5:6] == target_classes[None])
    pi, ti = valid.nonzero(as_tuple=True)
    candidates = sorted(((int(p), int(t), float(overlap[p, t])) for p, t in zip(pi, ti)),
                        key=lambda row: row[2], reverse=True)
    selected, used_predictions, used_targets = [], set(), set()
    for pred, target, score in candidates:
        if pred not in used_predictions and target not in used_targets:
            selected.append((pred, target, score))
            used_predictions.add(pred)
            used_targets.add(target)
    return selected


def test(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001,
         iou_thres=0.6, save_json=False, save_json_kpt=False, single_cls=False,
         augment=False, verbose=False, model=None, dataloader=None, save_dir=Path(''),
         save_txt=False, save_txt_tidl=False, save_hybrid=False, save_conf=False,
         plots=True, wandb_logger=None, compute_loss=None, half_precision=True,
         is_coco=False, opt=None, tidl_load=False, dump_img=False, kpt_label=False,
         flip_test=False):
    training = model is not None
    save_dir = Path(save_dir)
    if not training:
        if opt is None:
            raise ValueError('Standalone validation requires command-line options')
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
        model = attempt_load(weights, map_location=device)
    else:
        device = next(model.parameters()).device
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_txt or save_txt_tidl:
        (save_dir/'labels').mkdir(exist_ok=True)
    if isinstance(data, (str, Path)):
        with open(data) as handle:
            data = yaml.safe_load(handle)
    check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    head = model.model[-1]
    nkpt = int(getattr(head, 'nkpt', 0) or 0) if kpt_label else 0
    configured_nkpt = int(data.get('nkpt', data.get('kpt_shape', [nkpt])[0]))
    if kpt_label and nkpt != configured_nkpt:
        raise ValueError(f'Model has {nkpt} keypoints but data has {configured_nkpt}')
    keypoint_names = data.get('kpt_names', data.get('keypoint_names', model.yaml.get('kpt_names', [str(i) for i in range(nkpt)])))
    if isinstance(keypoint_names, dict):
        keypoint_names = keypoint_names.get(0, keypoint_names.get('0', []))
    if len(keypoint_names) != nkpt:
        keypoint_names = [str(i) for i in range(nkpt)]
    flip_idx = data.get('flip_idx', model.yaml.get('flip_idx', list(range(nkpt))))
    skeleton = data.get('skeleton', model.yaml.get('skeleton'))
    half = device.type != 'cpu' and half_precision
    model.half() if half else model.float()
    model.eval()
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, s=gs)
    if dataloader is None:
        task = opt.task if opt and opt.task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt,
                                      pad=0.5, rect=True, prefix=colorstr(f'{task}: '),
                                      tidl_load=tidl_load, kpt_label=kpt_label,
                                      nkpt=nkpt, flip_idx=flip_idx)[0]
    names = getattr(model, 'names', [str(i) for i in range(nc)])
    names = dict(enumerate(names)) if not isinstance(names, dict) else names
    confusion_matrix = ConfusionMatrix(nc=nc)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    loss = torch.zeros(5, device=device)
    stats, predictions_json = [], []
    seen, inference_time, nms_time = 0, 0.0, 0.0
    point_count = np.zeros(nkpt, dtype=float)
    matched_count = np.zeros(nkpt, dtype=float)
    point_error = np.zeros(nkpt, dtype=float)
    point_correct = np.zeros(nkpt, dtype=float)
    thresholds = np.arange(0.02, 0.201, 0.02)
    pck_correct = np.zeros(len(thresholds), dtype=float)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Validating boxes and pose')):
        img = img.to(device, non_blocking=True)
        img = (img.half() if half else img.float()) / 255
        targets = targets.to(device)
        nb, _, height, width = img.shape
        with torch.no_grad():
            start = time_synchronized()
            out, train_out = model(img, augment=augment)
            if not kpt_label:
                out = out[..., :5+nc]
            if flip_test:
                flipped, _ = model(img.flip(3), augment=False)
                flipped[..., 0] = width - flipped[..., 0]
                if nkpt:
                    points = flipped[..., 5+nc:].reshape(*flipped.shape[:-1], nkpt, 3)
                    points[..., 0] = width - points[..., 0]
                    flipped[..., 5+nc:] = points[..., flip_idx, :].reshape(*flipped.shape[:-1], 3*nkpt)
                out = torch.cat((out, flipped), 1)
            inference_time += time_synchronized() - start
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets, training=False)[1][:-1]
            pixel_targets = targets.clone()
            pixel_targets[:, 2:6] *= targets.new_tensor([width, height, width, height])
            if nkpt:
                pixel_targets[:, 6:] *= targets.new_tensor([width, height] * nkpt)
            hybrid = [pixel_targets[pixel_targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            start = time_synchronized()
            output = non_max_suppression(out, conf_thres, iou_thres, labels=hybrid,
                                         multi_label=True, agnostic=single_cls,
                                         kpt_label=kpt_label, nc=nc, nkpt=nkpt)
            nms_time += time_synchronized() - start
        for image_index, pred in enumerate(output):
            seen += 1
            path = Path(paths[image_index])
            labels = pixel_targets[pixel_targets[:, 0] == image_index, 1:].clone()
            target_classes = labels[:, 0]
            target_boxes = xywh2xyxy(labels[:, 1:5])
            scale_coords(img.shape[2:], target_boxes, shapes[image_index][0], shapes[image_index][1])
            target_points = labels[:, 5:].clone()
            if nkpt:
                scale_coords(img.shape[2:], target_points, shapes[image_index][0], shapes[image_index][1], kpt_label=True)
                target_points = target_points.reshape(-1, nkpt, 2)
                target_valid = (target_points != 0).any(2)
                point_count += target_valid.sum(0).cpu().numpy()
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img.shape[2:], predn[:, :4], shapes[image_index][0], shapes[image_index][1])
            if nkpt:
                scale_coords(img.shape[2:], predn[:, 6:], shapes[image_index][0], shapes[image_index][1], kpt_label=True, step=3)
            correct = torch.zeros((len(pred), len(iouv)), dtype=torch.bool, device=device)
            # Re-match at each IoU threshold for valid box AP in multi-animal scenes.
            for threshold_index, threshold in enumerate(iouv):
                for pi, ti, overlap in match_instances(predn, target_classes, target_boxes, float(threshold)):
                    correct[pi, threshold_index] = True
            if nkpt:
                for pi, ti, overlap in match_instances(predn, target_classes, target_boxes):
                    valid = target_valid[ti]
                    predicted = predn[pi, 6:].reshape(nkpt, 3)[:, :2]
                    error = torch.linalg.vector_norm(predicted - target_points[ti], dim=1)
                    diagonal = torch.linalg.vector_norm(target_boxes[ti, 2:] - target_boxes[ti, :2]).clamp_min(1e-9)
                    normalized_error = error / diagonal
                    mask = valid.cpu().numpy()
                    point_error += error.cpu().numpy() * mask
                    matched_count += mask
                    point_correct += ((normalized_error <= 0.1) & valid).cpu().numpy()
                    for index, threshold in enumerate(thresholds):
                        pck_correct[index] += ((normalized_error <= threshold) & valid).sum().item()
            stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), target_classes.cpu().numpy()))
            if plots:
                confusion_matrix.process_batch(predn[:, :6], torch.cat((target_classes[:, None], target_boxes), 1))
            if save_txt or save_txt_tidl:
                gain = predn.new_tensor([shapes[image_index][0][1], shapes[image_index][0][0]] * 2)
                lines = []
                for row in predn:
                    normalized_box = (xyxy2xywh(row[:4][None]) / gain)[0].tolist()
                    values = [int(row[5]), *normalized_box]
                    if nkpt:
                        points = row[6:].reshape(nkpt, 3).clone()
                        points[:, 0] /= shapes[image_index][0][1]
                        points[:, 1] /= shapes[image_index][0][0]
                        values.extend(points.flatten().tolist())
                    if save_conf:
                        values.append(float(row[4]))
                    lines.append(' '.join(f'{value:g}' for value in values))
                (save_dir/'labels'/(path.stem+'.txt')).write_text('\n'.join(lines)+'\n')
            if save_json or save_json_kpt:
                for row in predn:
                    box = xyxy2xywh(row[:4][None])[0]
                    box[:2] -= box[2:]/2
                    item=dict(image_id=path.stem, category_id=int(row[5]), score=float(row[4]), bbox=box.tolist())
                    if nkpt:
                        item['keypoints']=row[6:].tolist()
                    predictions_json.append(item)
        if plots and batch_i < 3:
            plot_images(img, targets, paths, save_dir/f'val_batch{batch_i}_labels.jpg', names,
                        kpt_label=kpt_label, skeleton=skeleton)
            plot_images(img, output_to_target(output), paths, save_dir/f'val_batch{batch_i}_pred.jpg', names,
                        kpt_label=kpt_label, steps=3, skeleton=skeleton)
    mp = mr = map50 = map75 = map90 = mean_ap = 0.0
    ap_class, ap = np.array([], dtype=int), np.zeros(0)
    if stats:
        tp, confidence, predicted_class, target_class = [np.concatenate(values, 0) for values in zip(*stats)]
        if tp.any():
            precision, recall, aps, f1, ap_class = ap_per_class(tp, confidence, predicted_class, target_class,
                                                              plot=plots, save_dir=save_dir, names=names)
            mp, mr = float(precision.mean()), float(recall.mean())
            map50, map75, map90 = [float(aps[:, i].mean()) for i in [0, 5, 8]]
            ap = aps.mean(1)
            mean_ap = float(ap.mean())
    per_point = point_correct / np.maximum(point_count, 1)
    pck = pck_correct / max(point_count.sum(), 1)
    error_px = float(point_error.sum()/max(matched_count.sum(), 1))
    report = dict(keypoints=keypoint_names, bbox_precision=mp, bbox_recall=mr, bbox_map50=map50,
                  bbox_map50_95=mean_ap, matched_keypoint_error_px=error_px,
                  per_keypoint_pck_at_0_1=dict(zip(keypoint_names, per_point.tolist())),
                  labelled_points=point_count.tolist(), matched_points=matched_count.tolist(),
                  pck_by_bbox_diagonal={f'{threshold:.2f}':float(value) for threshold,value in zip(thresholds,pck)},
                  pck_denominator='All labelled ground-truth points, including unmatched animals',
                  error_denominator='Labelled points on geometrically matched animals only')
    (save_dir/'pose_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Images {seen}; box P {mp:.4f}, R {mr:.4f}, AP50 {map50:.4f}, AP50:95 {mean_ap:.4f}; '
          f'keypoint error {error_px:.2f} px; PCK@0.1 {pck[4]:.4f}')
    if save_json or save_json_kpt:
        (save_dir/'predictions.json').write_text(json.dumps(predictions_json)+'\n')
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    model.float()
    maps = np.zeros(nc) + mean_ap
    for index, category in enumerate(ap_class):
        maps[category] = ap[index]
    times = tuple(value/max(seen,1)*1000 for value in [inference_time,nms_time,inference_time+nms_time]) + (imgsz,imgsz,batch_size)
    results = (mp,mr,map50,map75,map90,mean_ap,error_px,*per_point.tolist(),*pck.tolist(),
               *((loss.cpu()/max(len(dataloader),1)).tolist()))
    return results, maps, times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--tidl-load', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--dump-img', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--save-json-kpt', action='store_true', help='save a cocoapi-compatible JSON results file for key-points')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--kpt-label', action='store_true', help='Whether kpt-label is enabled or not')
    parser.add_argument('--flip-test', action='store_true', help='Whether to run flip_test or not')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_json_kpt |= opt.data.endswith('coco_kpts.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    # Dependencies are managed explicitly via requirements-rat.txt; never install
    # or update the user's environment as a side effect of evaluation.

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.save_json_kpt,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_txt_tidl=opt.save_txt_tidl,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf, 
			 opt=opt,
             tidl_load = opt.tidl_load,
             dump_img = opt.dump_img,
             kpt_label = opt.kpt_label,
             flip_test = opt.flip_test,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
