"""Export RGB + oracle 2D prompts using WildDet3D's annotation filtering."""
import argparse
from pathlib import Path

from core.io import read_json, write_json, source_signature
from scripts.wilddet3d_protocol import load_official, evaluator_kwargs, UPSTREAM_COMMIT


def exclude_images(data, file_paths):
    """Remove explicitly selected images and their GT from the scoring population."""
    if not isinstance(file_paths, list) or not file_paths or not all(isinstance(x, str) for x in file_paths):
        raise ValueError('Exclusions must be a nonempty JSON list of image file paths')
    paths = set(file_paths)
    known = {im['file_path'] for im in data['images']}
    if paths - known:
        raise ValueError(f'Unknown excluded image paths: {sorted(paths - known)}')
    removed = [im for im in data['images'] if im['file_path'] in paths]
    removed_ids = {im['id'] for im in removed}
    subset = {**data,
              'images': [im for im in data['images'] if im['id'] not in removed_ids],
              'annotations': [ann for ann in data['annotations'] if ann['image_id'] not in removed_ids]}
    if not subset['images']:
        raise ValueError('Exclusions leave no evaluation images')
    scope = {'type': 'explicit_image_subset', 'original_images': len(data['images']),
             'evaluated_images': len(subset['images']), 'excluded_images': len(removed),
             'excluded_annotations': len(data['annotations']) - len(subset['annotations']),
             'excluded_image_ids': sorted(removed_ids), 'excluded_file_paths': sorted(paths)}
    return subset, scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--annotation', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--benchmark', required=True,
                   help='stereo4d, in_the_wild, scannet, argoverse, or omni3d_<Subset>')
    p.add_argument('--output', required=True)
    p.add_argument('--exclude-image-paths', help='JSON list of explicitly excluded image paths')
    p.add_argument('--subset-annotation', help='Separate GT annotation output, required with exclusions')
    args = p.parse_args()
    if bool(args.exclude_image_paths) != bool(args.subset_annotation):
        p.error('--exclude-image-paths and --subset-annotation must be used together')
    annotation = args.annotation
    scope = None
    if args.exclude_image_paths:
        if Path(args.subset_annotation).resolve() == Path(annotation).resolve():
            p.error('Subset annotations must not overwrite the original annotations')
        subset, scope = exclude_images(read_json(annotation), read_json(args.exclude_image_paths))
        scope['original_annotation'] = source_signature(annotation)
        scope['exclusion_list'] = source_signature(args.exclude_image_paths)
        write_json(args.subset_annotation, subset)
        annotation = args.subset_annotation
    load_official(args.upstream)
    from wilddet3d.data.datasets.coco3d import COCO3D
    kw = evaluator_kwargs(args.benchmark, annotation)
    # Only InTheWild explicitly overrides the dataset loader's minimum height.
    min_height = 0.0 if args.benchmark == 'in_the_wild' else 0.0625
    api = COCO3D(annotation, list(kw['det_map']), min_height_thres=min_height)
    raw = read_json(annotation)
    samples = []
    for image_id in sorted(api.imgs):
        im = api.imgs[image_id]
        detections = []
        for a in api.imgToAnns[image_id]:
            if a['ignore'] or a['category_name'] in {'dontcare', 'ignore', 'void'}:
                continue
            x, y, w, h = a['bbox']
            detections.append({'annotation_id': a['id'], 'category_id': a['category_id'],
                               'category': a['category_name'], 'bbox_xyxy': [x, y, x+w, y+h]})
        samples.append({'image_id': image_id, 'file_path': im['file_path'],
                        'image_size': [im['width'], im['height']], 'detections': detections})
    write_json(args.output, {'protocol': 'wilddet3d_gt2d_rgb_only_v1', 'upstream_commit': UPSTREAM_COMMIT,
                            'benchmark': args.benchmark, 'annotation': source_signature(annotation),
                            'data_root': str(Path(args.data_root).resolve()), 'categories': raw['categories'],
                            'prompt_min_height': min_height, 'samples': samples,
                            **({'evaluation_scope': scope} if scope else {})})
    print(f'{args.benchmark}: {len(samples)} images, {sum(len(s["detections"]) for s in samples)} prompts')


if __name__ == '__main__':
    main()
