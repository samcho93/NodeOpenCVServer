#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundle example JSON flows with images from Images/ folder into ZIP project files.
Output: examples_zip/ folder with .zip files ready for Load.
"""
import os
import json
import zipfile
import uuid
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, 'examples')
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'examples_zip')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping: example filename -> list of image/video files for each image_read/video_read node (in order)
EXAMPLE_ASSETS = {
    '01_finger_counter.json':         [('video', 'ocean.mp4')],
    '02_fade_transition.json':        [('image', 'rose.jpg'), ('image', 'desert.jpg')],
    '03_motion_detection.json':       [('video', 'ocean.mp4'), ('video', 'ocean.mp4')],
    '04_color_tracking.json':         [('video', 'ocean.mp4')],
    '05_face_detection.json':         [('image', 'family.jpg')],
    '06_cartoon_effect.json':         [('image', 'cat.jpg')],
    '07_document_scanner.json':       [('image', 'transport.jpg')],
    '08_histogram_clahe.json':        [('image', 'wintry.jpg')],
    '09_watermark_blend.json':        [('image', 'snow.jpg'), ('image', 'tree.jpg')],
    '10_edge_art.json':               [('image', 'bird.jpg')],
    '11_noise_removal.json':          [('image', 'flamingo.jpg')],
    '12_image_negative.json':         [('image', 'rose.jpg')],
    '13_panorama_matching.json':      [('image', 'snow.jpg'), ('image', 'wintry.jpg')],
    '14_hough_line_detection.json':   [('image', 'transport.jpg')],
    '15_hough_circle_detection.json': [('image', 'plate.jpg')],
    '16_grabcut_segmentation.json':   [('image', 'bird.jpg')],
    '17_watershed_segmentation.json': [('image', 'hex.jpg')],
    '18_perspective_transform.json':  [('image', 'transport.jpg')],
    '19_flood_fill_coloring.json':    [('image', 'fairy.jpg')],
    '20_morphology_comparison.json':  [('image', '111.jpg')],
    '21_sharpen_compare.json':        [('image', 'flamingo.jpg')],
    '22_color_extraction.json':       [('image', 'rose.jpg')],
    '23_video_edge_output.json':      [('video', 'ocean.mp4')],
    '24_roi_focus.json':              [('image', 'cat.jpg')],
    '25_image_rotation_gallery.json': [('image', 'plane.png')],
    '26_threshold_comparison.json':   [('image', 'nape.jpg')],
    '27_contour_analysis.json':       [('image', 'hex.jpg')],
    '28_emboss_effect.json':          [('image', 'sparkler.jpg')],
    '29_feature_detection_compare.json': [('image', 'wine.jpg')],
    '30_custom_filter.json':          [('image', 'snow.jpg')],
}


def gen_id():
    return uuid.uuid4().hex[:8]


def bundle_example(json_filename, assets):
    json_path = os.path.join(EXAMPLES_DIR, json_filename)
    if not os.path.exists(json_path):
        print(f'  [SKIP] {json_filename} not found')
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        flow = json.load(f)

    # Find all image_read and video_read nodes in order
    input_nodes = []
    for node in flow.get('nodes', []):
        if node['type'] in ('image_read', 'video_read'):
            input_nodes.append(node)

    if len(input_nodes) != len(assets):
        # Handle case where same video is used for multiple nodes (e.g., motion detection)
        # or mismatch
        if len(assets) < len(input_nodes):
            print(f'  [WARN] {json_filename}: {len(input_nodes)} input nodes but {len(assets)} assets. Padding with last asset.')
            while len(assets) < len(input_nodes):
                assets.append(assets[-1])

    # Track used video ids to avoid duplicating same video file
    video_id_map = {}  # source_filename -> (vid_id, vid_path)

    zip_images = {}   # imageId -> image_path (for images)
    zip_videos = {}   # videoId -> video_path (for videos)

    for i, node in enumerate(input_nodes):
        if i >= len(assets):
            break
        asset_type, asset_file = assets[i]
        asset_path = os.path.join(IMAGES_DIR, asset_file)

        if not os.path.exists(asset_path):
            print(f'  [WARN] Asset not found: {asset_file}')
            continue

        file_id = gen_id()

        if asset_type == 'image' and node['type'] == 'image_read':
            node['properties']['imageId'] = file_id
            node['properties']['filename'] = asset_file
            node['label'] = asset_file
            zip_images[file_id] = asset_path

        elif asset_type == 'video' and node['type'] == 'video_read':
            # Reuse same video id if same file
            if asset_file in video_id_map:
                vid_id = video_id_map[asset_file][0]
            else:
                vid_id = file_id
                video_id_map[asset_file] = (vid_id, asset_path)
                zip_videos[vid_id] = asset_path

            node['properties']['imageId'] = vid_id
            node['properties']['filename'] = asset_file
            node['label'] = asset_file

        elif asset_type == 'video' and node['type'] == 'image_read':
            # Video asset assigned to image_read: extract first frame
            cap = cv2.VideoCapture(asset_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    node['properties']['imageId'] = file_id
                    node['properties']['filename'] = asset_file
                    node['label'] = asset_file
                    # Save temp frame
                    tmp_path = os.path.join(OUTPUT_DIR, f'_tmp_{file_id}.png')
                    cv2.imwrite(tmp_path, frame)
                    zip_images[file_id] = tmp_path

        elif asset_type == 'image' and node['type'] == 'video_read':
            # Image assigned to video_read: skip (shouldn't happen)
            print(f'  [WARN] Image assigned to video_read node in {json_filename}')

    # Create ZIP
    zip_name = json_filename.replace('.json', '.zip')
    zip_path = os.path.join(OUTPUT_DIR, zip_name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # flow.json
        zf.writestr('flow.json', json.dumps(flow, indent=2, ensure_ascii=False))

        # Images
        for img_id, img_path in zip_images.items():
            ext = os.path.splitext(img_path)[1]
            zf.write(img_path, f'images/{img_id}{ext}')

        # Videos
        for vid_id, vid_path in zip_videos.items():
            ext = os.path.splitext(vid_path)[1]
            zf.write(vid_path, f'videos/{vid_id}{ext}')

    # Clean up temp files
    for img_id, img_path in zip_images.items():
        if '_tmp_' in img_path and os.path.exists(img_path):
            os.remove(img_path)

    # Report
    size_kb = os.path.getsize(zip_path) / 1024
    n_img = len(zip_images)
    n_vid = len(zip_videos)
    parts = []
    if n_img: parts.append(f'{n_img} images')
    if n_vid: parts.append(f'{n_vid} videos')
    print(f'  [OK] {zip_name} ({size_kb:.0f} KB, {", ".join(parts)})')
    return True


def main():
    print(f'Bundling {len(EXAMPLE_ASSETS)} examples...')
    print(f'  Images folder: {IMAGES_DIR}')
    print(f'  Output folder: {OUTPUT_DIR}')
    print()

    success = 0
    for json_file, assets in EXAMPLE_ASSETS.items():
        if bundle_example(json_file, list(assets)):
            success += 1

    print(f'\nDone: {success}/{len(EXAMPLE_ASSETS)} ZIP files created in examples_zip/')


if __name__ == '__main__':
    main()
