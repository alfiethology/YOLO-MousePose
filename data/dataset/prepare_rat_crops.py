#!/usr/bin/env python3
"""Export neighbour-screened six-point rat pose crops and annotated preview sheets."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

from prepare_rat_dataset import KEYPOINTS, FLIP_IDX, EDGES, VIDEO, sha, write_json, yolo_row

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent


def box_of(shape):
    points = shape['points'][:4] if shape.get('shape_type') == 'skeleton' else shape['points']
    xs, ys = zip(*points)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.


def focal_match(focal, proposals, threshold):
    scores = [(iou(focal, row['box']), index) for index, row in enumerate(proposals)]
    if not scores:
        return None
    score, index = max(scores)
    return index if score >= threshold else None


def crop_bounds(shape, width, height, padding):
    x1, y1, x2, y2 = box_of(shape)
    # Retain any labelled landmarks outside the animal's box; do not clip them.
    for (x, y), visible in zip(shape['points'][4:], shape['pose']['visibility']):
        if visible:
            x1, y1, x2, y2 = min(x1,x), min(y1,y), max(x2,x), max(y2,y)
    return [max(0,math.floor(x1)-padding), max(0,math.floor(y1)-padding),
            min(width,math.ceil(x2)+padding), min(height,math.ceil(y2)+padding)]


def neighbour_hits(bounds, proposals, self_index, min_area):
    return [dict(item, overlap_pixels=intersection(bounds,item['box']))
            for index,item in enumerate(proposals)
            if index != self_index and intersection(bounds,item['box']) > min_area]


def csv_write(path, rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def preview_tile(full, record, shape):
    tile=Image.new('RGB',(600,300),'white');draw=ImageDraw.Draw(tile)
    colour='#008a20' if record['status']=='accepted' else '#ba2020'
    draw.rectangle((0,0,599,299),outline=colour,width=3)
    draw.text((8,6),f'{record["crop_id"]} / {record["split"]} / {record["status"]}',fill=colour)
    context=full.copy();cd=ImageDraw.Draw(context)
    for hit in record['neighbours']:
        cd.rectangle(hit['box'],outline='red',width=5)
    cd.rectangle(record['crop_bounds'],outline='yellow',width=5)
    cd.rectangle(box_of(shape),outline='lime',width=4)
    context.thumbnail((294,221));tile.paste(context,(5,37))
    left,top,right,bottom=record['crop_bounds'];crop=full.crop((left,top,right,bottom));pd=ImageDraw.Draw(crop)
    pd.rectangle([v for pair in [(box_of(shape)[0]-left,box_of(shape)[1]-top),(box_of(shape)[2]-left,box_of(shape)[3]-top)] for v in pair],outline='lime',width=2)
    points=[[x-left,y-top] for x,y in shape['points'][4:]];visibility=shape['pose']['visibility']
    for a,b in EDGES:
        if visibility[a] and visibility[b]:pd.line([tuple(points[a]),tuple(points[b])],fill='cyan',width=2)
    colours=['red','blue','orange','yellow','lime','magenta']
    for index,((x,y),visible) in enumerate(zip(points,visibility)):
        if visible:pd.ellipse((x-3,y-3,x+3,y+3),fill=colours[index])
    crop.thumbnail((288,221));tile.paste(crop,(306+(288-crop.width)//2,37+(221-crop.height)//2))
    draw.text((8,266),record['reason'][:95],fill='black')
    draw.text((8,282),Path(record['source_image']).stem[:92],fill='black')
    return tile


def make_sheets(output, tiles):
    previews=output/'previews';previews.mkdir(exist_ok=True)
    pages=[]
    for status in ['accepted','excluded']:
        selected=[tile for state,split,tile in tiles if state==status]
        for start in range(0,len(selected),12):
            batch=selected[start:start+12];sheet=Image.new('RGB',(1800,300*math.ceil(len(batch)/3)),'#dddddd')
            for index,tile in enumerate(batch):sheet.paste(tile,((index%3)*600,(index//3)*300))
            name=f'{status}_{start//12+1:02d}.jpg';sheet.save(previews/name,quality=92);pages.append(name)
    overview=[]
    rng=random.Random(42)
    for status in ['accepted','excluded']:
        for split,size in [('train',5),('val',4)]:
            pool=[tile for state,source_split,tile in tiles if state==status and source_split==split]
            overview.extend(rng.sample(pool,min(size,len(pool))))
    sheet=Image.new('RGB',(1800,300*math.ceil(len(overview)/3)),'#dddddd')
    for index,tile in enumerate(overview):sheet.paste(tile,((index%3)*600,(index//3)*300))
    sheet.save(output/'preview_overview.jpg',quality=93)
    return pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=PROJECT/'frames/ratcam1/labelled/pose_estimation')
    parser.add_argument('--master',type=Path,default=PROJECT/'frames/ratcam1/labelled/rat_bb_master/reviewed')
    parser.add_argument('--detector',type=Path,default=PROJECT/'runs/detect/ratcam1_bb_trials_80e_20261028/rat/t01_baseline/weights/best.pt')
    parser.add_argument('--output',type=Path,default=PROJECT/'models/datasets/pose/ratcam1_mousepose_rat6_crops_v1')
    parser.add_argument('--config',type=Path,default=REPO/'data/rat_crops_kpts.yaml')
    parser.add_argument('--device',default='0')
    parser.add_argument('--padding',type=int,default=10)
    parser.add_argument('--detector-conf',type=float,default=0.15)
    parser.add_argument('--match-iou',type=float,default=0.3)
    parser.add_argument('--max-neighbour-overlap',type=float,default=16.,help='exclude if another box intersects crop by more than this many square pixels')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    if args.padding < 0 or args.max_neighbour_overlap < 0 or not 0 < args.match_iou <= 1 or not 0 < args.detector_conf <= 1:
        parser.error('Invalid screening parameters')
    if not args.detector.is_file():raise FileNotFoundError(args.detector)
    from ultralytics import YOLO
    detector=YOLO(str(args.detector))
    if detector.task!='detect' or {str(v).lower() for v in detector.names.values()}!={'rat'}:
        raise ValueError('Expected a single-class BB rat detector')
    shared=PROJECT/'models/datasets/ratcam1_bb_master_split_v1/source_split.csv'
    with shared.open() as handle:recording_splits={row['video_group']:row['split'] for row in csv.DictReader(handle)}
    master_index={p.stem:p for p in args.master.glob('*.json')}
    source_records=[];hashes={};image_hashes=set();master_evidence=[]
    for annotation in sorted(args.source.rglob('*.json')):
        doc=json.loads(annotation.read_text());shapes=[s for s in doc.get('shapes',[]) if s.get('shape_type')=='skeleton']
        if not shapes:continue
        image=annotation.with_suffix('.jpg');h=sha(image)
        if h in image_hashes:raise ValueError(f'Duplicate source image {image}')
        image_hashes.add(h);hashes[str(image)]=h;hashes[str(annotation)]=sha(annotation)
        match=VIDEO.search(image.stem)
        if not match:raise ValueError(f'Missing recording {image}')
        recording=match.group()
        if recording not in recording_splits:
            value=int(hashlib.sha256(f'{args.seed}:{recording}'.encode()).hexdigest()[:16],16)/16**16
            recording_splits[recording]='val' if value<0.2 else 'train'
        source_records.append((annotation,image,doc,shapes,recording))
    if not source_records:raise ValueError('No pose annotations')
    for split in ['train','val']:
        for folder in ['images','labels']:(output/folder/split).mkdir(parents=True)
    all_records=[];tiles=[];counts=Counter();accepted_counts=Counter();bbox_counts=Counter();detections_log=[]
    for frame_number,(annotation,image,doc,shapes,recording) in enumerate(source_records,1):
        with Image.open(image) as im:full=im.convert('RGB')
        width,height=full.size
        if (width,height)!=(doc['imageWidth'],doc['imageHeight']):raise ValueError(f'Dimensions disagree {image}')
        proposals=[];master_used=False;master_score=None
        candidate=master_index.get(image.stem)
        if candidate:
            other_image=candidate.with_suffix('.jpg');other_doc=json.loads(candidate.read_text())
            if (other_doc['imageWidth'],other_doc['imageHeight'])==(width,height):
                same=sha(other_image)==hashes[str(image)]
                if same:master_score=1.
                else:
                    a=cv2.cvtColor(np.asarray(full),cv2.COLOR_RGB2GRAY);b=cv2.imread(str(other_image),cv2.IMREAD_GRAYSCALE)
                    if b is not None and b.shape==a.shape:master_score=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0])
                if master_score is not None and master_score>=0.995:
                    proposals=[dict(box=box_of(s),evidence='reviewed_master',confidence=1.) for s in other_doc['shapes']]
                    master_used=True
                    master_evidence.extend([dict(path=str(candidate),sha256=sha(candidate)),dict(path=str(other_image),sha256=sha(other_image))])
        if not master_used:
            result=detector.predict(cv2.cvtColor(np.asarray(full),cv2.COLOR_RGB2BGR),conf=args.detector_conf,iou=0.45,imgsz=960,
                                    device=args.device,max_det=16,agnostic_nms=True,verbose=False)[0]
            if result.boxes is not None:
                proposals=[dict(box=box.tolist(),confidence=float(conf),evidence='rat_detector')
                           for box,conf in zip(result.boxes.xyxy.cpu(),result.boxes.conf.cpu())]
        detections_log.append(dict(source_image=str(image),master_used=master_used,master_pixel_match=master_score,boxes=proposals))
        pose_proposals=[dict(box=box_of(s),evidence='pose_annotation',confidence=1.,pose_index=i) for i,s in enumerate(shapes)]
        for shape_index,shape in enumerate(shapes):
            # Validate source geometry/schema/visibility even for excluded examples.
            yolo_row(shape,width,height)
            focal=box_of(shape);bounds=crop_bounds(shape,width,height,args.padding)
            matched=focal_match(focal,proposals,args.match_iou)
            neighbours=neighbour_hits(bounds,pose_proposals,shape_index,args.max_neighbour_overlap)
            neighbours+=neighbour_hits(bounds,proposals,matched,args.max_neighbour_overlap)
            reasons=[]
            if shape.get('flags',{}).get('exclude_from_pose_training'):reasons.append('existing_pose_exclusion')
            if not any(shape['pose']['visibility']):reasons.append('no_labelled_landmarks')
            if matched is None:reasons.append('focal_not_matched')
            if neighbours:reasons.append('neighbour_enters_crop')
            status='excluded' if reasons else 'accepted';split=recording_splits[recording]
            crop_id=f'{image.stem}__pose{shape_index:02d}'
            record=dict(crop_id=crop_id,status=status,reason='; '.join(reasons) if reasons else 'isolated_crop',
                        split=split,recording=recording,source_image=str(image),source_annotation=str(annotation),
                        shape_index=shape_index,crop_bounds=bounds,focal_bbox=focal,
                        box_evidence='reviewed_master' if master_used else 'rat_detector',
                        focal_match_iou=iou(focal,proposals[matched]['box']) if matched is not None else None,
                        labelled_landmarks=sum(v>0 for v in shape['pose']['visibility']),
                        source_pose_review_incomplete=doc.get('flags',{}).get('pose_review_incomplete',False),
                        neighbours=neighbours,output_image=None)
            counts[status]+=1
            if status=='accepted':
                left,top,right,bottom=bounds;cropped=full.crop(bounds)
                local=deepcopy(shape);local['points']=[[x-left,y-top] for x,y in shape['points']]
                row=yolo_row(local,cropped.width,cropped.height)
                destination=output/'images'/split/(crop_id+'.jpg');cropped.save(destination,quality=97,subsampling=0)
                (output/'labels'/split/(crop_id+'.txt')).write_text(row+'\n')
                record['output_image']=str(destination.relative_to(output))
                accepted_counts[split]+=1;bbox_counts[record['box_evidence']]+=1
            all_records.append(record);tiles.append((status,split,preview_tile(full,record,shape)))
        if frame_number%10==0:print(f'{frame_number}/{len(source_records)} frames; {counts["accepted"]} accepted, {counts["excluded"]} excluded',flush=True)
    if not all(accepted_counts[s]>0 for s in ['train','val']):raise ValueError(f'Empty crop split: {accepted_counts}')
    pages=make_sheets(output,tiles)
    write_json(output/'screening_records.json',all_records)
    write_json(output/'frame_box_evidence.json',detections_log)
    csv_rows=[]
    for r in all_records:
        csv_rows.append({k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()})
    csv_write(output/'screening_manifest.csv',csv_rows)
    csv_write(output/'manifest.csv',[r for r in csv_rows if r['status']=='accepted'])
    write_json(output/'source_hashes.json',hashes)
    write_json(output/'master_evidence_hashes.json',master_evidence)
    accepted=[r for r in all_records if r['status']=='accepted']
    summary=dict(status='exported; automatically neighbour-screened pose crops',source_frames=len(source_records),
                 source_skeletons=len(all_records),accepted_crops=dict(accepted_counts),excluded_crops=counts['excluded'],
                 exclusion_reasons=dict(Counter(reason for r in all_records if r['status']=='excluded' for reason in r['reason'].split('; '))),
                 accepted_box_evidence=dict(bbox_counts),recordings={s:len({r['recording'] for r in accepted if r['split']==s}) for s in ['train','val']},
                 crop_padding=args.padding,detector=str(args.detector.resolve()),detector_sha256=sha(args.detector),
                 detector_confidence=args.detector_conf,focal_match_min_iou=args.match_iou,
                 neighbour_overlap_max_square_pixels=args.max_neighbour_overlap,
                 preview_pages=len(pages),source_annotations_changed=False,
                 split_policy='Reuse shared rat/ID recording split; deterministic assignment for other recordings',
                 screening_limitations='Detector misses and bounding-box approximations remain possible; automatic screening is not human completeness certification')
    config=dict(train=str(output/'images/train'),val=str(output/'images/val'),nc=1,names=['rat'],nkpt=6,kpt_shape=[6,3],
                keypoint_names=KEYPOINTS,kpt_names=KEYPOINTS,flip_idx=FLIP_IDX,skeleton=EDGES,
                kpt_oks_sigmas=[1/6]*6,review_required=False,automatically_neighbour_screened=True,
                preparation_manifest=str(output/'manifest.csv'),screening_records=str(output/'screening_records.json'))
    yaml_text='# Crop-based rat pose dataset; see screening reports and previews.\n'+yaml.safe_dump(config,sort_keys=False)
    (output/'data.yaml').write_text(yaml_text);args.config.write_text(yaml_text)
    write_json(output/'summary.json',summary)
    (output/'README.md').write_text(
        '# Neighbour-screened RatCam1 pose crops\n\n'
        'Open preview_overview.jpg first, then previews/accepted_*.jpg and previews/excluded_*.jpg '
        'for every decision. Each tile shows full-frame context (left) and the pose crop (right). '
        'Yellow is the proposed crop, green the focal BB, red any neighbouring BB. Coloured landmarks '
        'and cyan edges show labelled pose points. Excluded examples are previews only, not training images.\n\n'
        'One accepted rat per crop, 10-pixel padding by default. Crop bounds include all labelled '
        'landmarks even if outside the focal BB. Unlabelled points remain visibility 0; labelled '
        'occluded points remain 1. Original images and JSON are unchanged.\n\n'
        'Neighbour evidence uses the reviewed master BB annotations when the source image is '
        'identical or strongly pixel-registered (correlation >=0.995); other frames use the generic '
        'single-class BB detector. Other pose annotations, including explicitly excluded poses, '
        'always count as neighbouring animals. The focal animal must match a master/detector BB '
        'at IoU >=0.3. Any other box intersecting the actual crop by more than 16 square pixels '
        'excludes that crop. Existing per-pose exclusion flags also exclude only that focal crop. '
        'This is deliberately conservative; overlapping boxes may cause false exclusions. '
        'Detector misses remain possible. Review the previews before a long run.\n\n'
        'The old full-frame completeness flag does not automatically exclude isolated crops. '
        'review_required=false here means the automatic crop screening completed, not that a human '
        'reviewed all crop previews. Low-quality or uncertain crops can be removed as image/TXT pairs '
        'or corrected at source and re-exported to a new dataset.\n\n'
        'Train using YOLO-MousePose/train_rat.py --data YOLO-MousePose/data/rat_crops_kpts.yaml '
        '--epochs 30 --name rat6_isolated_crops_30e. This is a functionality-compatible dataset '
        'for crop inference after a generic rat detector. Keypoint order: snout, left_ear, right_ear, '
        'neck_base, body_centre, tail_base. Matching uses the same source-recording assignment as '
        'the generic and identity exports where available; no recording spans both crop splits.\n')
    html='<!doctype html><meta charset="utf-8"><title>Pose crop preview sheets</title><h1>Pose crop screening</h1><p>Green: accepted. Red: held out. Left: frame context. Right: proposed crop. All decisions are in screening_manifest.csv.</p>'
    html+='<p><a href="preview_overview.jpg">Overview sheet</a></p>'
    for page in pages:html+=f'<h2>{page}</h2><a href="previews/{page}"><img src="previews/{page}" style="max-width:100%"></a>'
    (output/'preview_index.html').write_text(html)
    for name,expected in hashes.items():
        if sha(Path(name))!=expected:raise ValueError(f'Source changed: {name}')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
