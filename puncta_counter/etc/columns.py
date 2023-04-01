# Objects
# # ellipse_cols
# # nuclei_cols
# # puncta_cols


ellipse_cols = ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]

nuclei_cols = [
    # index cols
    'image_number',
    'nuclei_object_number',
    'path_name_tif',
    'file_name_tif',
    
    # square areas
    'center_x',
    'center_y',
    'bounding_box_min_x',
    'bounding_box_max_x',
    'bounding_box_min_y',
    'bounding_box_max_y',  # useful for selecting frames
    'bounding_box_area',  # area = (max_x-min_x)*(max_y-min_y)  # useful for images
    
    # size measures
    'orientation',  # The angle in degrees [-90, 90] between x-axis and major_axis
    'major_axis_length',  # "diameter" a
    'minor_axis_length',  # "diameter" b
    'area',  # area < pi*(a/2)*(b/2), actual area of the object
    'convex_area',  # convex_area > pi*(a/2)*(b/2), area within bounding box
    'perimeter',  # ~ 2*pi*np.sqrt(((a/2)**2+(b/2)**2)/2)  # perimeter of bounding box
    
    # measures of eccentricity
    'eccentricity',  # np.sqrt(1-(b**2/a**2))
    'form_factor',  # 4*pi*area/perimeter**2, equals 1 for a perfectly circular object.
    'compactness',  # 1/form_factor. The mean squared distance of the object’s pixels from the centroid divided by the area.
    
    
    # ----------------------------------------------------------------------
    # Don't need
    
    # ratios
    # 'equivalent_diameter',  # area = pi*(equivalent_diameter/2)**2, diameter of circle with the same area as the object
    # 'extent',  #  area/box_area
    # 'solidity',  # area/convex_area
    # 'euler_number',  # this is always 1

    # max distance between tangent lines
    # 'max_feret_diameter',  # similar to major_axis_length
    # 'min_feret_diameter',  # similar to minor_axis_length
    
    # distances to outside the radius
    # 'mean_radius',
    # 'median_radius', 
    # 'max_radius',
    
    # redundant
    # 'location_center_x',  # same as center_x
    # 'location_center_y',  # same as center_y
    # 'location_center_z',  # always 0
    # 'number_object_number',  # same as object_number 
]


puncta_cols = [
    "image_number", "puncta_object_number", "parent_manual_nuclei",  # index_cols
    
    # square areas
    "center_x",
    "center_y",
    "bounding_box_min_x",
    "bounding_box_max_x",
    "bounding_box_min_y",
    "bounding_box_max_y",    
    "bounding_box_area",
    
    # size measures
    "orientation",
    "major_axis_length",
    "minor_axis_length",
    "area",
    "convex_area",
    "perimeter",
    
    # measures of eccentricity
    "eccentricity",
    "form_factor",
    "compactness",
    
    # intensities
    "integrated_intensity",
    "min_intensity",
    "max_intensity",
    "mean_intensity",
    "median_intensity",
    
    # edge intensities?
    "edge_integrated_intensity",
    "edge_min_intensity",
    "edge_max_intensity",
    "edge_mean_intensity",
    
    
    # ----------------------------------------------------------------------
    # Don't need
    
    # ratios
    # "equivalent_diameter",
    # "solidity",
    # "extent",
    # "euler_number",
    
    # max distance between tangent lines
    # "min_feret_diameter",
    # "max_feret_diameter",
    
    # distances to outside the radius
    # "mean_radius",
    # "median_radius",
    # "maximum_radius",
    
    # intensities
    # "mad_intensity",  # median absolute deviation (MAD) of the intensities within the object
    # "std_intensity",
    # "mass_displacement",  # distance between the centers of gravity in the gray-level representation of the object and the binary representation of the object.
    # "lower_quartile_intensity",    
    # "upper_quartile_intensity,
    # "edge_std_intensity",
    
    # redundant
    # "location_center_x",  # same as center_x
    # "location_center_y",  # same as center_y
    # "location_center_z",  # always 0
    # "location_center_mass_intensity_x",  # similar to center_x with more digits
    # "location_center_mass_intensity_y",  # similar to center_x with more digits
    # "location_center_mass_intensity_z",  # always 0
    # "location_max_intensity_x",  # similar to center_x, but rounded
    # "location_max_intensity_y",  # similar to center_y, but rounded
    # "location_max_intensity_z",  # always 0
    # "number_object_number",  # same as object_number 
]
