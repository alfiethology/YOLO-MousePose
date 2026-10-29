#!/usr/bin/env python3
"""Export focal pose crops retaining neighbours, with all known intersecting targets."""
import argparse,csv,hashlib,json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import yaml
from PIL import Image
from prepare_rat_dataset import KEYPOINTS,FLIP_IDX,EDGES,VIDEO,sha,write_json,yolo_row
from prepare_rat_crops import box_of,crop_bounds,intersection,iou,preview_tile,make_sheets,csv_write
REPO=Path(__file__).resolve().parents[2]
PROJECT=REPO.parent

def local_target(shape,bounds,pose_enabled=True):
    left,top,right,bottom=bounds
    b=box_of(shape);b=[max(left,b[0])-left,max(top,b[1])-top,min(right,b[2])-left,min(bottom,b[3])-top]
    if b[2]-b[0]<2 or b[3]-b[1]<2:return None
    points=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
    visibility=[0]*6;landmarks=[[0,0] for _ in range(6)]
    if pose_enabled and shape.get('shape_type')=='skeleton':
        for k,((x,y),v) in enumerate(zip(shape['points'][4:],shape['pose']['visibility'])):
            if v and left<=x<=right and top<=y<=bottom:landmarks[k]=[x-left,y-top];visibility[k]=v
    return dict(shape_type='skeleton',label='rat',points=points+landmarks,pose=dict(keypoints=KEYPOINTS,visibility=visibility))

def whole_neighbour_bounds(focal, targets, width, height, padding):
    """Expand to include complete known neighbours, repeating as the context grows."""
    bounds = crop_bounds(focal, width, height, padding)
    while True:
        previous = list(bounds)
        for target in targets:
            if intersection(bounds, box_of(target)) <= 0:
                continue
            if target.get('shape_type') == 'skeleton':
                candidate = crop_bounds(target, width, height, padding)
            else:
                import math
                a,b,c,d = box_of(target)
                candidate = [max(0,math.floor(a)-padding),max(0,math.floor(b)-padding),
                             min(width,math.ceil(c)+padding),min(height,math.ceil(d)+padding)]
            bounds = [min(bounds[0],candidate[0]),min(bounds[1],candidate[1]),
                      max(bounds[2],candidate[2]),max(bounds[3],candidate[3])]
        if bounds == previous:
            return bounds


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=PROJECT/'frames/ratcam1/labelled/pose_estimation')
    ap.add_argument('--output',type=Path,default=PROJECT/'models/datasets/pose/ratcam1_mousepose_rat6_crowded_crops_v3')
    ap.add_argument('--config',type=Path,default=REPO/'data/rat_crops_kpts.yaml')
    ap.add_argument('--padding',type=int,default=10)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if args.padding<0:ap.error('padding must be nonnegative')
    with (PROJECT/'models/datasets/ratcam1_bb_master_split_v1/source_split.csv').open() as f:splits={r['video_group']:r['split'] for r in csv.DictReader(f)}
    # Reuse previously audited frame boxes for older annotations lacking box guides.
    evidence_path=PROJECT/'models/datasets/pose/ratcam1_mousepose_rat6_crops_v1/frame_box_evidence.json'
    evidence={Path(r['source_image']).stem:r['boxes'] for r in json.loads(evidence_path.read_text())} if evidence_path.exists() else {}
    for split in ['train','val']:
        for kind in ['images','labels']:(out/kind/split).mkdir(parents=True)
    records=[];tiles=[];hashes={};seen=set();stats=Counter();frames=[]
    for ann in sorted(args.source.rglob('*.json')):
        doc=json.loads(ann.read_text());poses=[s for s in doc.get('shapes',[]) if s.get('shape_type')=='skeleton']
        if not poses:continue
        image=ann.with_suffix('.jpg');h=sha(image)
        if h in seen:raise ValueError(f'Duplicate image {image}')
        seen.add(h);hashes[str(image)]=h;hashes[str(ann)]=sha(ann)
        match=VIDEO.search(image.stem)
        if not match:raise ValueError(image)
        recording=match.group()
        if recording not in splits:splits[recording]='val' if int(hashlib.sha256(f'42:{recording}'.encode()).hexdigest()[:16],16)/16**16<.2 else 'train'
        split=splits[recording];frames.append(dict(source_image=str(image),recording=recording,split=split))
        with Image.open(image) as im:full=im.convert('RGB')
        if full.size!=(doc['imageWidth'],doc['imageHeight']):raise ValueError(image)
        for p in poses:yolo_row(p,*full.size)
        targets=list(poses)
        guides=[s for s in doc['shapes'] if s.get('shape_type')=='rectangle']
        if not guides:
            for e in evidence.get(image.stem,[]):
                x1,y1,x2,y2=e['box'];guides.append(dict(shape_type='rectangle',points=[[x1,y1],[x2,y2]],label='rat'))
        for guide in guides:
            if not any(iou(box_of(guide),box_of(s))>=.3 for s in targets):targets.append(guide)
        emitted_bounds = set()
        for idx,p in enumerate(poses):
            reasons=[]
            if p.get('flags',{}).get('exclude_from_pose_training'):reasons.append('existing_pose_exclusion')
            if not any(p['pose']['visibility']):reasons.append('no_labelled_landmarks')
            bounds=whole_neighbour_bounds(p,targets,*full.size,args.padding);crop_id=f'{image.stem}__pose{idx:02d}'
            rows=[];neighbours=[];bbox_only=0
            for target in targets:
                if intersection(bounds,box_of(target))<=0:continue
                enabled=not target.get('flags',{}).get('exclude_from_pose_training',False)
                local=local_target(target,bounds,enabled)
                if local is None:continue
                rows.append(yolo_row(local,bounds[2]-bounds[0],bounds[3]-bounds[1]))
                bbox_only+=not any(local['pose']['visibility'])
                if target is not p:neighbours.append(dict(box=box_of(target)))
            if not reasons and tuple(bounds) in emitted_bounds:reasons.append('duplicate_group_crop')
            rec=dict(crop_id=crop_id,status='excluded' if reasons else 'accepted',reason='; '.join(reasons) or 'neighbours_retained',split=split,recording=recording,source_image=str(image),source_annotation=str(ann),shape_index=idx,crop_bounds=bounds,neighbours=neighbours,targets=len(rows),box_only_targets=bbox_only,output_image=None)
            if not reasons:
                emitted_bounds.add(tuple(bounds))
                dest=out/'images'/split/(crop_id+'.jpg');full.crop(bounds).save(dest,quality=97,subsampling=0)
                (out/'labels'/split/(crop_id+'.txt')).write_text('\n'.join(rows)+'\n');rec['output_image']=str(dest.relative_to(out))
                stats[split+'_crops']+=1;stats[split+'_targets']+=len(rows);stats[split+'_box_only_targets']+=bbox_only;stats['crops_with_neighbours']+=bool(neighbours);stats['expanded_to_include_whole_neighbours']+=bounds!=crop_bounds(p,*full.size,args.padding)
            records.append(rec);tiles.append((rec['status'],split,preview_tile(full,rec,p)))
    assert stats['train_crops'] and stats['val_crops']
    pages=make_sheets(out,tiles);write_json(out/'crop_records.json',records);write_json(out/'source_hashes.json',hashes);csv_write(out/'source_split.csv',frames)
    csv_write(out/'manifest.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in records if r['status']=='accepted'])
    config=dict(train=str(out/'images/train'),val=str(out/'images/val'),nc=1,names=['rat'],nkpt=6,kpt_shape=[6,3],keypoint_names=KEYPOINTS,kpt_names=KEYPOINTS,flip_idx=FLIP_IDX,skeleton=EDGES,kpt_oks_sigmas=[1/6]*6,review_required=False,automatically_neighbour_screened=False,neighbours_retained=True,preparation_manifest=str(out/'manifest.csv'))
    text=yaml.safe_dump(config,sort_keys=False);(out/'data.yaml').write_text(text);args.config.write_text(text)
    summary=dict(source_frames=len(frames),source_skeletons=len(records),**stats,excluded_focal_crops=sum(r['status']=='excluded' for r in records),exclusion_reasons=dict(Counter(r['reason'] for r in records if r['status']=='excluded')),padding=args.padding,preview_pages=len(pages),split_policy='Shared rat/ID recording groups, seed 42 deterministic fallback',source_files_changed=False)
    write_json(out/'summary.json',summary)
    (out/'README.md').write_text('# Rat pose crops retaining neighbours\n\nEach eligible skeleton produces one crop with 10-pixel padding. The crop expands iteratively to include whole known neighbouring rat boxes and their labelled landmarks, avoiding fragment-only detection targets. Nearby rats never cause exclusion. Every intersecting saved skeleton contributes a clipped box and its in-crop available landmarks. Box-only neighbours and explicitly excluded poses contribute detection boxes with all landmarks unavailable, not guessed points. Explicit exclusions cannot be focal crops. Existing guide boxes (or earlier audited frame boxes for older frames) supplement skeletons, with IoU-based duplicate suppression. Unknown animals lacking any saved box or skeleton may remain unannotated.\n\nRecording groups follow the shared rat/ID split. Crops from one frame/recording never cross splits. Previews show focal landmarks; TXT labels also contain known neighbours. Validation PCK covers available landmarks only; box-only animals can still count for detection. Identical group crop bounds from the same frame are exported only once. No sources were modified. Review preview_overview.jpg and previews/ before training.\n')
    for p,h in hashes.items():assert sha(Path(p))==h,p
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
