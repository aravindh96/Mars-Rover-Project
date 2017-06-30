import numpy as np
import cv2
import math

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def obstacle_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def rock_thresh(img, rgb_thresh):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,1] > rgb_thresh[2]) \
                & (img[:,:,2] < rgb_thresh[3])
    color_select[above_thresh] = 1
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away fromvertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO
    # Convert yaw to radians
    rad_angle=yaw*(math.pi/180)
    # Apply a rotation
    xpix_rotated = (xpix*math.cos(rad_angle)) - (ypix*math.sin(rad_angle))
    ypix_rotated = (xpix*math.sin(rad_angle)) + (ypix*math.cos(rad_angle))

    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = xpos + (xpix_rot/scale)
    ypix_translated = ypos + (ypix_rot/scale)
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    image=Rover.img

    # 1) Define source and destination points for perspective transform
    dst_size=5
    bottom_offset=6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped=perspect_transform(image,source,destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    #1 Navigable terrain
    terrain_threshed=color_thresh(warped,(160,160,160))
    rock_threshed=rock_thresh(warped,(75,250,50,50))
    obstacle_threshed=obstacle_thresh(warped,(160,160,160))
    # 4) Update Rover.vision_image (this will be displayed on left side of scre[en)
    Rover.vision_image[:,:,0]=terrain_threshed*255
    Rover.vision_image[:,:,1]=rock_threshed*255
    Rover.vision_image[:,:,2]=obstacle_threshed*255
    Rover.warped_image=warped

        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    terrain_rover_x,terrain_rover_y=rover_coords(terrain_threshed)
    rock_rover_x,rock_rover_y=rover_coords(rock_threshed)
    obstacle_rover_x,obstacle_rover_y=rover_coords(obstacle_threshed)
    # 6) Convert rover-centric pixel values to world coordinates
    terrain_world_x,terrain_world_y= pix_to_world(terrain_rover_x,terrain_rover_y,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)
    rock_world_x,rock_world_y= pix_to_world(rock_rover_x,rock_rover_y,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)
    obstacle_world_x,obstacle_world_y= pix_to_world(obstacle_rover_x,obstacle_rover_y,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],10)

    if (Rover.roll < 5):
        if  (Rover.pitch <5):
            if (Rover.mode=='forward'):
                    # 7) Update Rover worldmap (to be displayed on right side of screen)
                Rover.worldmap[terrain_world_y,terrain_world_x,2]=255
                Rover.worldmap[rock_world_y,rock_world_x,1]=255
                Rover.worldmap[obstacle_world_y,obstacle_world_x,0]=255
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist,angles= to_polar_coords(terrain_rover_x,terrain_rover_y)
    # Update Rover pixel distances and angles
    Rover.nav_dists = dist
    Rover.nav_angles = angles




    return Rover
