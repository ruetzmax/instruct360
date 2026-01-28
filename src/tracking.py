from src.operations2d import get_2d_bounding_boxes, bounding_boxes_to_image_chunks
from src.operations3d import get_3d_bounding_boxes, adjust_centers_by_chunk_rotation, get_box_meshes

def get_bounding_boxes_for_class(image, class_name, threshold_2d=0.35, threshold_3d=0.3):
    bounding_boxes_2d = get_2d_bounding_boxes(image, class_name, threshold=threshold_2d)
    
    image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d)
    
    all_bb_centers = []
    all_bb_dimensions = []
    all_bb_poses = []
    for chunk in image_chunks:
        bb_centers, bb_dimensions, bb_poses = get_3d_bounding_boxes(chunk, class_name, threshold=threshold_3d)
        bb_centers_adjusted = adjust_centers_by_chunk_rotation(bb_centers, chunk)
        all_bb_centers.extend(bb_centers_adjusted)
        all_bb_dimensions.extend(bb_dimensions)
        all_bb_poses.extend(bb_poses)
        
    return all_bb_centers, all_bb_dimensions, all_bb_poses