from typing import List, Tuple, Optional

def calculate_centroid(bbox: List[int]) -> Tuple[float, float]:
    """
    Calculate (y_center, x_center) from [ymin, xmin, ymax, xmax].
    """
    if not bbox or len(bbox) != 4:
        return (0.0, 0.0)
    y_center = (bbox[0] + bbox[2]) / 2.0
    x_center = (bbox[1] + bbox[3]) / 2.0
    return (y_center, x_center)

def calculate_relative_direction(src_bbox: Optional[List[int]], dst_bbox: Optional[List[int]]) -> str:
    """
    Calculate the relative direction of Source FROM Destination.
    (e.g., if Source is above Destination, returns 'Top')
    
    This answers: "From the perspective of the Destination node, where is the Source node coming from?"
    """
    if not src_bbox or not dst_bbox:
        return "Unknown"
    
    src_y, src_x = calculate_centroid(src_bbox)
    dst_y, dst_x = calculate_centroid(dst_bbox)
    
    dy = src_y - dst_y  # negative if src is above (assuming y=0 at top)
    dx = src_x - dst_x  # negative if src is left
    
    # Simple quadrant check or angle based
    # y increases downwards in images usually (0,0 is top-left)
    
    # Determine primary axis
    if abs(dy) > abs(dx):
        # Vertical relation
        if dy < 0:
            return "Top"    # Source is above (y is smaller)
        else:
            return "Bottom" # Source is below
    else:
        # Horizontal relation
        if dx < 0:
            return "Left"   # Source is left
        else:
            return "Right"  # Source is right

