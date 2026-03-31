#!/usr/bin/env python

# Copyright (c) 2019-2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# This work is licensed under the terms of the MIT license.

"""
Combined script: Generate Traffic + 2D/3D Bounding Boxes + Auto-Flying Drone

Drone Controls:
    The drone automatically follows the road network via Waypoints.
    R            : Toggle recording images and bounding boxes
    3            : Visualize bounding boxes in 3D
    2            : Visualize bounding boxes in 2D
    ESC          : Quit
"""

import carla
import socket
from contextlib import closing


#Spawning actors
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor

import argparse
import logging
import json
import random
import queue
import pygame
import numpy as np
import time
from math import radians
from PIL import Image
from pygame.locals import K_ESCAPE, K_2, K_3, K_r
import cv2
import os
from datetime import datetime
import h5py
# --- Bounding Box Constants & Functions ---
EDGES = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

SEMANTIC_MAP = {
    0: ('unlabelled', (0,0,0)), 1: ('road', (128,64,0)), 2: ('sidewalk', (244,35,232)),
    3: ('building', (70,70,70)), 4: ('wall', (102,102,156)), 5: ('fence', (190,153,153)),
    6: ('pole', (153,153,153)), 7: ('traffic light', (250,170,30)), 8: ('traffic sign', (220,220,0)), 
    9: ('vegetation', (107,142,35)), 10: ('terrain', (152,251,152)), 11: ('sky', (70,130,180)), 
    12: ('pedestrian', (220,20,60)), 13: ('rider', (255,0,0)), 14: ('car', (0,0,142)), 
    15: ('truck', (0,0,70)), 16: ('bus', (0,60,100)), 17: ('train', (0,80,100)), 
    18: ('motorcycle', (0,0,230)), 19: ('bicycle', (119,11,32)), 20: ('static', (110,190,160)), 
    21: ('dynamic', (170,120,50)), 22: ('other', (55,90,80)), 23: ('water', (45,60,150)), 
    24: ('road line', (157,234,50)), 25: ('ground', (81,0,81)), 26: ('bridge', (150,100,100)), 
    27: ('rail track', (230,150,140)), 28: ('guard rail', (180,165,180))
}

def get_open_port():
    """Asks the OS to allocate a free port and returns the port number"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def decode_instance_segmentation(img_rgba: np.ndarray):
    semantic_labels = img_rgba[..., 2]
    actor_ids = img_rgba[..., 1].astype(np.uint16) + (img_rgba[..., 0].astype(np.uint16) << 8)
    return semantic_labels, actor_ids

def bbox_2d_for_actor(actor, actor_ids: np.ndarray, semantic_labels: np.ndarray):
    actor_id_16bit = actor.id & 0xFFFF
    mask = (actor_ids == actor_id_16bit)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return {'actor_id': actor.id,
            'semantic_label': actor.semantic_tags[0] if len(actor.semantic_tags) > 0 else 0,
            'bbox_2d': (xmin, ymin, xmax, ymax)}

def bbox_3d_for_actor(actor, camera_bp, camera):
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    npc_bbox_loc = actor.get_transform().location + actor.bounding_box.location
    npc_bbox_loc_arr = np.array([npc_bbox_loc.x, npc_bbox_loc.y, npc_bbox_loc.z, 1.0])
    npc_loc_ego = np.dot(world_2_camera, npc_bbox_loc_arr)
    npc_loc_ego_space = carla.Location(x=float(npc_loc_ego[0]), y=float(npc_loc_ego[1]), z=float(npc_loc_ego[2]))

    verts = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    projection = []
    
    cam_forward_vec = camera.get_transform().get_forward_vector()
    
    for edge in EDGES:
        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

        if not p1_in_canvas and not p2_in_canvas:
            continue

        ray0 = verts[edge[0]] - camera.get_transform().location
        ray1 = verts[edge[1]] - camera.get_transform().location

        if not (cam_forward_vec.dot(ray0) > 0):
            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
        if not (cam_forward_vec.dot(ray1) > 0):
            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
        
        projection.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

    return {'actor_id': actor.id,
            'semantic_label': actor.semantic_tags[0] if len(actor.semantic_tags) > 0 else 0,
            'bbox_3d': {
                'center': {'x': npc_loc_ego_space.x, 'y': npc_loc_ego_space.y, 'z': npc_loc_ego_space.z},
                'dimensions': {'length': actor.bounding_box.extent.x*2, 'width': actor.bounding_box.extent.y*2, 'height': actor.bounding_box.extent.z*2},
                'rotation_yaw': radians(actor.get_transform().rotation.yaw - camera.get_transform().rotation.yaw)
            },
            'projection': projection}

def visualize_2d_bboxes(surface, img, bboxes, font):
    rgb_img = img[:, :, :3][:, :, ::-1] 
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img[..., 0:3], (1,0,2)))
    surface.blit(frame_surface, (0, 0))
    
    for item in bboxes:
        bbox = item['2d']
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(v) for v in bbox['bbox_2d']]
            label = SEMANTIC_MAP[bbox['semantic_label']][0]
            color = SEMANTIC_MAP[bbox['semantic_label']][1]
            pygame.draw.rect(surface, color, pygame.Rect(xmin, ymin, xmax-xmin, ymax-ymin), 2)
            # Use the passed font object
            text_surface = font.render(label, True, (255,255,255), color) 
            text_rect = text_surface.get_rect(topleft=(xmin, ymin-20))
            surface.blit(text_surface, text_rect)

def visualize_3d_bboxes(surface, img, bboxes, font):
    rgb_img = img[:, :, :3][:, :, ::-1] 
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img[..., 0:3], (1,0,2)))
    surface.blit(frame_surface, (0, 0))
    for item in bboxes:
        bbox = item['3d']
        color = SEMANTIC_MAP[bbox['semantic_label']][1]
        n = 0
        mean_x = 0
        mean_y = 0
        for line in bbox['projection']:
            mean_x += line[0]
            mean_y += line[1]
            n += 1
            pygame.draw.line(surface, color, (line[0], line[1]), (line[2],line[3]), 2)
        if n > 0:
            mean_x /= n
            mean_y /= n
            # Use the passed font object
            text_surface = font.render(SEMANTIC_MAP[bbox['semantic_label']][0], True, (255,255,255), color)
            text_rect = text_surface.get_rect(topleft=(mean_x, mean_y))
            surface.blit(text_surface, text_rect)

# Convert both to pygame surfaces, draw bboxes on each
def draw_bboxes_on_frame(frame_rgb, frame_bboxes, display_3d, font):
    """Draw bboxes onto a numpy RGB frame and return it."""
    surf = pygame.Surface((frame_rgb.shape[1], frame_rgb.shape[0]))
    pygame.surfarray.blit_array(surf, np.transpose(frame_rgb, (1, 0, 2)))
    
    if display_3d:
        for item in frame_bboxes:
            bbox = item['3d']
            color = SEMANTIC_MAP[bbox['semantic_label']][1]
            for line in bbox['projection']:
                pygame.draw.line(surf, color, (line[0], line[1]), (line[2], line[3]), 2)
    else:
        for item in frame_bboxes:
            bbox = item['2d']
            if bbox:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox['bbox_2d']]
                color = SEMANTIC_MAP[bbox['semantic_label']][1]
                pygame.draw.rect(surf, color, pygame.Rect(xmin, ymin, xmax-xmin, ymax-ymin), 2)
    
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))


def get_actor_velocity_dict(actor):
    v = actor.get_velocity()
    return {'x': v.x, 'y': v.y, 'z': v.z}

def vehicle_light_state_to_dict(vehicle):
    if not hasattr(vehicle, 'get_light_state'):
        return {}
    state = vehicle.get_light_state()
    return {
        "position": bool(state & carla.VehicleLightState.Position),
        "low_beam": bool(state & carla.VehicleLightState.LowBeam),
        "high_beam": bool(state & carla.VehicleLightState.HighBeam),
        "brake": bool(state & carla.VehicleLightState.Brake),
        "reverse": bool(state & carla.VehicleLightState.Reverse),
        "left_blinker": bool(state & carla.VehicleLightState.LeftBlinker),
        "right_blinker": bool(state & carla.VehicleLightState.RightBlinker),
    }

# --- Traffic Generation Helpers ---
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all": return bps
    if len(bps) == 1: return bps
    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            return [x for x in bps if int(x.get_attribute('generation')) == int_generation]
        else:
            return []
    except:
        return []

# --- Main Script ---
def main():
    argparser = argparse.ArgumentParser(description='CARLA Traffic & Bounding Boxes with Auto Drone')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-n', '--number-of-vehicles', metavar='N', default=30, type=int, help='Number of vehicles (default: 30)')
    argparser.add_argument('-w', '--number-of-walkers', metavar='W', default=0, type=int, help='Number of walkers (default: 0)')
    argparser.add_argument('-d', '--distance', metavar='D', default=50, type=int, help='Actor BB rendering distance threshold')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--tm-port', metavar='P', default=8268, type=int, help='Port to communicate with TM (default: 8268)')
    argparser.add_argument('--seed', metavar='S', type=int, help='Set random device seed')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    pygame.init()
    pygame.font.init()
    default_font = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()
    pygame.display.set_caption("Auto Drone View & Bounding Boxes")
    display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    random.seed(args.seed if args.seed is not None else int(time.time()))

    vehicles_list = []
    walkers_list = []
    all_id = []
    
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    dynamic_tm_port = get_open_port()
    logging.info(f"Using dynamic Traffic Manager port: {dynamic_tm_port}")
    traffic_manager = client.get_trafficmanager(dynamic_tm_port)
    traffic_manager.global_percentage_speed_difference(30.0) # Drive 30% under the speed limit
    traffic_manager.set_global_distance_to_leading_vehicle(2.5) # Maintain a 2.5-meter gap
    traffic_manager.set_synchronous_mode(True)
    carla_map = world.get_map()

    # Enable Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    try:
        # --------------
        # Spawn Traffic
        # --------------
        blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
        blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*', '2')
        
        spawn_points = carla_map.get_spawn_points()
        random.shuffle(spawn_points)

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles: break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        
        for response in client.apply_batch_sync(batch, True):
            if not response.error: vehicles_list.append(response.actor_id)

        # Spawn Walkers
        spawn_points_w = []
        for i in range(args.number_of_walkers):
            loc = world.get_random_location_from_navigation()
            if loc != None: spawn_points_w.append(carla.Transform(loc))

        batch = []
        walker_speed = []
        for spawn_point in spawn_points_w:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'): walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
            
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error: walkers_list[i]["con"] = results[i].actor_id

        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
            
        all_actors = world.get_actors(all_id)
        world.tick()

        for i in range(0, len(all_id), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        logging.info(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers.")

        # --------------
        # Spawn Auto-Drone (Camera) RGB
        # --------------
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(args.width))
        rgb_bp.set_attribute('image_size_y', str(args.height))
        # Initialize Waypoint Tracking
        start_transform = random.choice(spawn_points)
        current_waypoint = carla_map.get_waypoint(start_transform.location)
        
        # Start the drone 10 meters in the air, pitched slightly down
        drone_altitude = 20.0
        drone_pitch = -15.0
        drone_speed = 1.0 # Meters per tick
        
        drone_transform = carla.Transform(
            current_waypoint.transform.location + carla.Location(z=drone_altitude), 
            carla.Rotation(pitch=drone_pitch, yaw=current_waypoint.transform.rotation.yaw)
        )
        drone_camera = world.spawn_actor(rgb_bp, drone_transform)

        inst_camera_bp = bp_lib.find('sensor.camera.instance_segmentation')
        inst_camera_bp.set_attribute('image_size_x', str(args.width))
        inst_camera_bp.set_attribute('image_size_y', str(args.height))
        inst_camera = world.spawn_actor(inst_camera_bp, carla.Transform(), attach_to=drone_camera)

        dvs_bp = bp_lib.find('sensor.camera.dvs')
        dvs_bp.set_attribute('image_size_x', str(args.width))
        dvs_bp.set_attribute('image_size_y', str(args.height))
        dvs_bp.set_attribute('positive_threshold', '0.3')
        dvs_bp.set_attribute('negative_threshold', '0.3')
        dvs_bp.set_attribute('sigma_positive_threshold', '0.0')
        dvs_bp.set_attribute('sigma_negative_threshold', '0.0')
        dvs_bp.set_attribute('refractory_period', '0')
        dvs_camera = world.spawn_actor(dvs_bp, carla.Transform(), attach_to=drone_camera)

        image_queue = queue.Queue()
        drone_camera.listen(image_queue.put)

        inst_queue = queue.Queue()
        inst_camera.listen(inst_queue.put)

        dvs_queue = queue.Queue()
        dvs_camera.listen(dvs_queue.put)

        # Generate a unique string for this run (e.g., "20260326_223204")
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_dir = f"_out/{run_stamp}_images"
        vid_dir = f"_out/{run_stamp}_video"
        
        # Safely create the directories if they don't exist
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(vid_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_rgb = cv2.VideoWriter(f'{vid_dir}/rgb_output.mp4', fourcc, 20.0, (args.width, args.height))
        video_out_dvs = cv2.VideoWriter(f'{vid_dir}/dvs_output.mp4', fourcc, 20.0, (args.width, args.height))
        combined_video = cv2.VideoWriter(f'{vid_dir}/combined.mp4', fourcc, 20.0, (args.width * 2, args.height))

        h5_path = f'{vid_dir}/data.h5'
        h5_file = h5py.File(h5_path, 'w')

        h5_events = h5_file.create_dataset('events/x',   shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=True)
        h5_y      = h5_file.create_dataset('events/y',   shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=True)
        h5_t      = h5_file.create_dataset('events/t',   shape=(0,), maxshape=(None,), dtype=np.int64,  chunks=True)
        h5_pol    = h5_file.create_dataset('events/pol', shape=(0,), maxshape=(None,), dtype=np.bool_,  chunks=True)
        h5_frames = h5_file.create_dataset('frame_index', shape=(0, 2), maxshape=(None, 2), dtype=np.int64, chunks=True)

        h5_file.attrs['width']  = args.width
        h5_file.attrs['height'] = args.height
        h5_file.attrs['sensor'] = 'dvs'

        event_cursor = 0

        start_time = None
        
        # Loop State
        record = True
        display_3d = True
        run_simulation = True
        recorded_frames = 0
        target_frames = 30*20 # 30 seconds at 20 FPS
        while run_simulation:
            # 1. Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_simulation = False
                elif event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE: run_simulation = False
                    elif event.key == K_r: record = not record
                    elif event.key == K_2: display_3d = False
                    elif event.key == K_3: display_3d = True

            # 2. Process Auto-Pilot Waypoint Following
            # Ask CARLA for the next waypoint at the specified distance
            next_waypoints = current_waypoint.next(drone_speed)
            
            # If an intersection offers multiple paths, just pick the first one
            if len(next_waypoints) > 0:
                current_waypoint = next_waypoints[0]

            # Update the camera transform
            new_loc = current_waypoint.transform.location + carla.Location(z=drone_altitude)
            new_rot = carla.Rotation(pitch=drone_pitch, yaw=current_waypoint.transform.rotation.yaw)
            drone_camera.set_transform(carla.Transform(new_loc, new_rot))

            # 3. Tick Simulation
            world.tick()
            snapshot = world.get_snapshot()
            
            if start_time is None:
                start_time = snapshot.timestamp.elapsed_seconds

            json_frame_data = {
                'frame_id': snapshot.frame,
                'timestamp': snapshot.timestamp.elapsed_seconds,
                'objects': [] 
            }

            try:
                # 4. Read Sensors
                image = image_queue.get(timeout=2.0)
                dvs_data = dvs_queue.get(timeout=2.0)
                inst_seg_image = inst_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: No image received from camera sensor.")
                continue
            
            
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            inst_seg = np.reshape(np.copy(inst_seg_image.raw_data), (inst_seg_image.height, inst_seg_image.width, 4))

            events = np.frombuffer(dvs_data.raw_data, dtype=np.dtype([
                ('x', np.uint16),
                ('y', np.uint16),
                ('t', np.float64),
                ('pol', np.bool_)
            ]))

            dvs_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

            if len(events) > 0:
                pos = events[events['pol'] == True]
                neg = events[events['pol'] == False]
                dvs_frame[pos['y'], pos['x']] = [0, 255, 0]  # Green for positive events
                dvs_frame[neg['y'], neg['x']] = [255, 0, 0]  # Red for negative events
            semantic_labels, actor_ids = decode_instance_segmentation(inst_seg)

            frame_bboxes = []

            # 5. Calculate Bounding Boxes
            for npc in world.get_actors().filter('*'):
                if npc.id != drone_camera.id and (npc.type_id.startswith('vehicle') or npc.type_id.startswith('walker')):
                    dist = npc.get_transform().location.distance(drone_camera.get_transform().location)

                    if dist < args.distance:
                        forward_vec = drone_camera.get_transform().get_forward_vector()
                        inter_actor_vec = npc.get_transform().location - drone_camera.get_transform().location

                        if forward_vec.dot(inter_actor_vec) > 0:
                            npc_bbox_2d = bbox_2d_for_actor(npc, actor_ids, semantic_labels)
                            npc_bbox_3d = bbox_3d_for_actor(npc, rgb_bp, drone_camera)
                            frame_bboxes.append({'3d': npc_bbox_3d, '2d': npc_bbox_2d})

                            json_frame_data['objects'].append({
                                'id': npc.id,
                                'class': SEMANTIC_MAP[npc.semantic_tags[0]][0] if len(npc.semantic_tags) > 0 else "unknown",
                                'blueprint_id': npc.type_id,
                                'velocity': get_actor_velocity_dict(npc),
                                'bbox_3d': npc_bbox_3d['bbox_3d'],
                                'bbox_2d': {
                                    'xmin': int(npc_bbox_2d['bbox_2d'][0]),
                                    'ymin': int(npc_bbox_2d['bbox_2d'][1]),
                                    'xmax': int(npc_bbox_2d['bbox_2d'][2]),
                                    'ymax': int(npc_bbox_2d['bbox_2d'][3]),
                                } if npc_bbox_2d else None,
                                'light_state': vehicle_light_state_to_dict(npc) if npc.type_id.startswith('vehicle') else None
                            })
            rgb_with_bbox = draw_bboxes_on_frame(img, frame_bboxes, display_3d, default_font)
            dvs_with_bbox = draw_bboxes_on_frame(dvs_frame, frame_bboxes, display_3d, default_font)

            rgb_surf = pygame.surfarray.make_surface(np.transpose(rgb_with_bbox, (1, 0, 2)))
            dvs_surf = pygame.surfarray.make_surface(np.transpose(dvs_with_bbox, (1, 0, 2)))

            # 6. Render to Pygame
            display.fill((0,0,0))
            display.blit(rgb_surf, (0, 0))
            display.blit(dvs_surf, (args.width, 0))
            pygame.display.flip()
            clock.tick(60)
            

            if record:
                raw = pygame.surfarray.array3d(display)
                raw = np.transpose(raw, (1, 0, 2))
                
                # Save to the new image directory
                Image.fromarray(rgb_with_bbox).save(f'{img_dir}/rgb_{snapshot.frame:08d}.png')
                Image.fromarray(dvs_with_bbox).save(f'{img_dir}/dvs_{snapshot.frame:08d}.png')
                video_out_rgb.write(cv2.cvtColor(rgb_with_bbox, cv2.COLOR_RGB2BGR))
                video_out_dvs.write(cv2.cvtColor(dvs_with_bbox, cv2.COLOR_RGB2BGR))
                combined = np.hstack([
                    cv2.cvtColor(rgb_with_bbox, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(dvs_with_bbox, cv2.COLOR_RGB2BGR)
                ])
                combined_video.write(combined)
                
                if len(events) > 0:
                    n = len(events)
                    h5_events.resize(event_cursor + n, axis=0)
                    h5_y.resize(event_cursor + n, axis=0)
                    h5_t.resize(event_cursor + n, axis=0)
                    h5_pol.resize(event_cursor + n, axis=0)
                    
                    h5_events[event_cursor:event_cursor + n] = events['x']
                    h5_y     [event_cursor:event_cursor + n] = events['y']
                    h5_t     [event_cursor:event_cursor + n] = events['t']
                    h5_pol   [event_cursor:event_cursor + n] = events['pol']
                    
                    h5_events[event_cursor:event_cursor + n] = events['x']
                    h5_y     [event_cursor:event_cursor + n] = events['y']
                    h5_t     [event_cursor:event_cursor + n] = events['t']
                    h5_pol   [event_cursor:event_cursor + n] = events['pol']

                    frame_idx = np.array([[event_cursor, event_cursor + n]], dtype=np.int64)
                    h5_frames.resize(h5_frames.shape[0] + 1, axis=0)
                    h5_frames[-1] = frame_idx

                    event_cursor += n
                # Save JSON to the image directory
                with open(f"{img_dir}/{snapshot.frame}.json", 'w') as f:
                    json.dump(json_frame_data, f)
                recorded_frames += 1

                if recorded_frames >= target_frames:
                    print(f"\n[Timer] Recorded {recorded_frames} frames. Stopping recording...")
                    run_simulation = False
                    
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        print('\nCleaning up actors...')
        
        # Notice the quotes around 'video_out'
        if 'video_out' in locals():
            video_out_dvs.release()
            video_out_rgb.release()
            print('Video saved successfully.')
        if 'h5_file' in locals():
            h5_file.close()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        if 'traffic_manager' in locals():
            traffic_manager.set_synchronous_mode(False)
        # Destroy sensors (drone)
        if 'drone_camera' in locals():
            drone_camera.stop()
            drone_camera.destroy()
        if 'inst_camera' in locals():
            inst_camera.stop()
            inst_camera.destroy()

        # Stop walkers and destroy all traffic
        all_actors = world.get_actors(all_id)
        for i in range(0, len(all_id), 2):
            if all_actors[i] is not None:
                all_actors[i].stop()

        client.apply_batch([DestroyActor(x) for x in vehicles_list])
        client.apply_batch([DestroyActor(x) for x in all_id])
        
        # --- NEW: Force the server to process the destruction ---
        world.tick()
        
        # Reset Synchronous Settings
        settings = world.get_settings()
        settings.synchronous_mode = False 
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)

        pygame.quit()
        print('Done.')

if __name__ == '__main__':
    print('Controls:')
    print('  The drone automatically follows the road network via Waypoints.')
    print('  R            : Toggle recording images as PNG and bounding boxes as JSON')
    print('  3            : View bounding boxes in 3D')
    print('  2            : View bounding boxes in 2D')
    print('  ESC          : Quit')
    main()
