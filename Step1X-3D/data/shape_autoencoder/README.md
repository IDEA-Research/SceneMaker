keys in .npz file:
    ['surface', 'sharp_surface', 'volume_rand_points', 'near_surface_points', 'bounds']
detail information:
    surface: [N, 6], the position and normal of surface points of the shape
    sharp_surface: [N, 6], the position and normal of sharp surface points of the shape
    volume_rand_points: [N, 4], the random points and SDF value in the volume within the bounding box of the shape
    near_surface_points: [N, 4], the random points and SDF value in the volume within the bounding box of the shape
    bounds: float, descript the bounding box of the shape