# Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

import pdb


def create_palette():
    # Define a palette with 24 colors for labels 0-23 (example colors)
    palette = [
        0,
        0,
        0,  # Label 0 (black)
        255,
        0,
        0,  # Label 1 (red)
        0,
        255,
        0,  # Label 2 (green)
        0,
        0,
        255,  # Label 3 (blue)
        255,
        255,
        0,  # Label 4 (yellow)
        255,
        0,
        255,  # Label 5 (magenta)
        0,
        255,
        255,  # Label 6 (cyan)
        128,
        0,
        0,  # Label 7 (dark red)
        0,
        128,
        0,  # Label 8 (dark green)
        0,
        0,
        128,  # Label 9 (dark blue)
        128,
        128,
        0,  # Label 10
        128,
        0,
        128,  # Label 11
        0,
        128,
        128,  # Label 12
        64,
        0,
        0,  # Label 13
        0,
        64,
        0,  # Label 14
        0,
        0,
        64,  # Label 15
        64,
        64,
        0,  # Label 16
        64,
        0,
        64,  # Label 17
        0,
        64,
        64,  # Label 18
        192,
        192,
        192,  # Label 19 (light gray)
        128,
        128,
        128,  # Label 20 (gray)
        255,
        165,
        0,  # Label 21 (orange)
        75,
        0,
        130,  # Label 22 (indigo)
        238,
        130,
        238,  # Label 23 (violet)
    ]
    # Extend the palette to have 768 values (256 * 3)
    palette.extend([0] * (768 - len(palette)))
    return palette


PALETTE = create_palette()


# Result Utils
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: Optional[float] = None
    label: Optional[str] = None
    box: Optional[BoundingBox] = None
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


# Utils
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# Post-processing Utils
def generate_colored_segmentation(label_image):
    # Create a PIL Image from the label image (assuming it's a 2D numpy array)
    label_image_pil = Image.fromarray(label_image.astype(np.uint8), mode="P")

    # Apply the palette to the image
    palette = create_palette()
    label_image_pil.putpalette(palette)

    return label_image_pil


def plot_segmentation(image, detections):
    seg_map = np.zeros(image.size[::-1], dtype=np.uint8)
    for i, detection in enumerate(detections):
        mask = detection.mask
        seg_map[mask > 0] = i + 1
    seg_map_pil = generate_colored_segmentation(seg_map)
    return seg_map_pil


# Grounded SAM
def prepare_model(
    device: str = "cuda",
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
):
    detector_id = (
        detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    )
    object_detector = pipeline(
        model=detector_id, task="zero-shot-object-detection", device=device
    )

    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    processor = AutoProcessor.from_pretrained(segmenter_id)
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)

    return object_detector, processor, segmentator


def detect(
    object_detector: Any,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(
    processor: Any,
    segmentator: Any,
    image: Image.Image,
    boxes: Optional[List[List[List[float]]]] = None,
    detection_results: Optional[List[Dict[str, Any]]] = None,
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    if detection_results is None and boxes is None:
        raise ValueError(
            "Either detection_results or detection_boxes must be provided."
        )

    if boxes is None:
        boxes = get_boxes(detection_results)

    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(
        segmentator.device, segmentator.dtype
    )

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    if detection_results is None:
        detection_results = [DetectionResult() for _ in masks]

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    object_detector,
    processor,
    segmentator,
    image: Union[Image.Image, str],
    labels: Union[str, List[str]],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
) -> Tuple[np.ndarray, List[DetectionResult], Image.Image]:
    if isinstance(image, str):
        image = load_image(image)
    if isinstance(labels, str):
        labels = labels.split(",")

    detections = detect(object_detector, image, labels, threshold)
    detections = segment(processor, segmentator, image, detection_results=detections, polygon_refinement=polygon_refinement)

    seg_map_pil = plot_segmentation(image, detections)

    return np.array(image), detections, seg_map_pil

def grounding_sam(image_path, labels, output_path, threshold=0.3, detector_id="IDEA-Research/grounding-dino-base", segmenter_id="facebook/sam-vit-base", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    object_detector, processor, segmentator = prepare_model(
        device=device, detector_id=detector_id, segmenter_id=segmenter_id
    )

    image_array, detections, seg_map_pil = grounded_segmentation(
        object_detector,
        processor,
        segmentator,
        image=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
    )

    os.makedirs(output_path, exist_ok=True)
    
    # save whole image
    seg_map_pil.save(os.path.join(output_path, "segmentation.png"))
    Image.fromarray(image_array).save(os.path.join(output_path, "scene_image.png"))
    
    # save every masks and images
    mask_path = os.path.join(output_path, "masks")
    masked_image_path = os.path.join(output_path, "masked_images")
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(masked_image_path, exist_ok=True)
    for i, detection in enumerate(detections):
        label = detection.label.replace(".", "").replace(" ", "_")
        mask = detection.mask
        mask_pil = Image.fromarray(mask)
        mask_pil.save(os.path.join(mask_path, f"mask_{i}_{label}.png"))
        masked_image = np.array(image_array)
        masked_image[mask == 0] = 255
        masked_image_pil = Image.fromarray(masked_image)
        masked_image_pil.save(os.path.join(masked_image_path, f"masked_image_{i}_{label}.png"))
    
    # save detection labels
    with open(os.path.join(output_path, "detections.txt"), "w") as f:
        for i, detection in enumerate(detections):
            label = detection.label.replace(".", "").replace(" ", "_")
            f.write(f"{label}\n")

    return image_array, detections, seg_map_pil
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detector_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter_id", type=str, default="facebook/sam-vit-huge")
    args = parser.parse_args()
    
    image_array, detections, seg_map_pil = grounding_sam(detector_id=args.detector_id, segmenter_id=args.segmenter_id, image_path=args.image_path, \
        labels=args.labels, output_path=args.output_path, threshold=args.threshold)

    
