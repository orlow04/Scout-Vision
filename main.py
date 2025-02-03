from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():

    # Carregar Vídeo
    video_path = 'input_data/08fd33_0.mp4'
    frames = read_video(video_path)
    
    # Inicializar Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # Captar posição dos objetos
    tracker.add_position_to_tracks(tracks)

    # Estimador do movimento da câmera
    # camera_movement_estimator = CameraMovementEstimator(frames[0])
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames,read_from_stub=True,stub_path='stubs/camera_movement_stubs.pkl')
    # camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # # Transformador de visão
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolar posição da bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # # Estimador de velocidade e distância
    # speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Atribuir jogador a time
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    # Atribuir posse de bola a jogador
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    # Demonstrar tracks de n jogadores
    output_video_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    #Demonstrar movimento da câmera
    #output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Demonstrar velocidade e distância
    # speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # Salvar Vídeo
    save_video(output_video_frames, 'output_data/video_output.avi')

if __name__ == '__main__':
    main()