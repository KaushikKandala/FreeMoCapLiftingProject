from pathlib import Path
import numpy as np

from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import MediapipeModelInfo

def calculate_spherical_angles(human: Human):
    body_3d_xyz = human.body.xyz
    spine_vector = body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['proximal'] - body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['distal']
    spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)
    spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])
    spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9))

    return spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude

def name_of_function(x:int,y:int):
    z = x+y

    
    return z

path_to_recording = Path(r"C:\Users\Matthis Lab\skellycam_data\recordings\2025-09-30_16-06-46_GMT-4_OKK_back_cams_good_lift_trial1")

path_to_data = path_to_recording/"output_data"/"mediapipe_skeleton_3d.npy"
print(path_to_data)

np.load(path_to_data)

actual_data = np.load(path_to_data) #loads into a numpy array 

print(actual_data.shape) #shape is in frames, markers, dimensions

actual_data[0,5,:] #loads the first frame, 5th marker, gives you the 3d coordinates of that 

human = Human.from_tracked_points_numpy_array(
    name = 'human', 
    model_info=MediapipeModelInfo(),
    tracked_points_numpy_array=np.load(path_to_data)
)

print(human)
print(human.body) #gives you only the body
print(human.body.xyz) #3d data for body

num_frames = human.body.xyz.num_frames

spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude = calculate_spherical_angles(
    human=human
) #polar is your back tilt

print(spine_vector_polar)
z = name_of_function(x = 5, y = 10)
print(z)


f =2 