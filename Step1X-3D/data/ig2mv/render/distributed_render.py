import json
import multiprocessing
import subprocess
from dataclasses import dataclass
from typing import Optional
import os
import megfile
import boto3

import argparse

parser = argparse.ArgumentParser(description='distributed rendering')

parser.add_argument('--workers_per_gpu', type=int,
                    help='number of workers per gpu.')
parser.add_argument('--input_models_path', type=str,
                    help='Path to a json file containing a list of 3D object files.')
parser.add_argument('--download_from_s3', type=bool, default=False,
                    help='Whether to download the obj from S3.')
parser.add_argument('--upload_to_s3', type=bool, default=False,
                    help='Whether to upload the rendered images to S3.')
parser.add_argument('--num_gpus', type=int, default=-1,
                    help='number of gpus to use. -1 means all available gpus.')
parser.add_argument('--gpu_list',nargs='+', type=int, 
                    help='the avalaible gpus')

parser.add_argument('--start_i', type=int, default=0,
                    help='the index of first object to be rendered.')

parser.add_argument('--end_i', type=int, default=-1,
                    help='the index of the last object to be rendered.')

parser.add_argument('--objaverse_root', type=str, default='s3://shared-3d/objaverse-XL/hf-objaverse-v1/glbs/',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--save_folder', type=str, default=None,
                    help='Path to a json file containing a list of 3D object files.')

args = parser.parse_args()

def check_task_finish(render_dir):
    flag = True
    files_type = ["color", 'normal', 'depth']
    VIEWS = [0, 1, 2, 3, 4, 5]
    if os.path.exists(render_dir):
        for t in files_type:
            for view_index in VIEWS:
                view_index = "%04d" % view_index
                if t=='depth':
                    fpath = os.path.join(render_dir, f'{t}_{view_index}.exr')
                else:
                    fpath = os.path.join(render_dir, f'{t}_{view_index}.webp')
                if not os.path.exists(fpath):
                    flag = False
    else:
        flag = False

    return flag


def megfile_download_obj(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    style = object_url.split("/")[-1].split(".")[1]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.{style}" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.{style}")
    # megfile the file and put it in local_path
    os.makedirs("tmp-objects", exist_ok=True)
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    megfile.s3.s3_download(f'{object_url}', tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    download_from_s3: bool,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        obj_render_path = os.path.join(args.save_folder, item.split('/')[-1].split('.')[0])
        if check_task_finish(obj_render_path):
            queue.task_done()
            print('========', item, 'rendered', '========')
            continue
        else:
            os.makedirs(obj_render_path, exist_ok = True)

        # Perform some operation on the item
        if download_from_s3:
            item = megfile_download_obj(item)

        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"python multi-view-render.py "
            f" --model_path {item}"
            f" --save_dir {args.save_folder}"
        )
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, args.gpu_list[gpu_i], args.download_from_s3)
            )
            process.daemon = True
            process.start()
        
    # Add items to the queue
    if args.input_models_path is not None:
        with open(args.input_models_path, "r") as f:
            model_paths = json.load(f)

    args.end_i = len(model_paths) if args.end_i > len(model_paths) else args.end_i

    for item in model_paths[args.start_i:args.end_i]:
        obj_path = os.path.join(args.objaverse_root, item)
        queue.put(obj_path)

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)