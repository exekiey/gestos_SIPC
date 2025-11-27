import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import pygame
import pymunk
import math
# import re


model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None

tips_id = [4,8,12,16,20]



def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global detection_result
  detection_result = result


def draw_landmarks_on_image(rgb_image, detection_result):

  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
  

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

#--------------------------------------------------------------------------------------------------------------------------

# Configuración de Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

# Configuración de Pymunk
space = pymunk.Space()
space.gravity = (0, 0)  # Sin gravedad, para mover libremente el objeto

# Crear un círculo en Pymunk que se moverá con la mano
body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
body.position = (320, 240)  # Posición inicial en el centro
circle = pymunk.Circle(body, 20)
space.add(body, circle)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result, num_hands = 2)

import pygame
import numpy as np

def play_tone(freq: float):
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 32767 * np.sin(2 * np.pi * freq * t)
    wave = wave.astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    sound = pygame.sndarray.make_sound(stereo_wave)
    sound.play(loops=-1)
    return sound

def value_to_note_frequency(value: float) -> float:
    value = max(0.0, min(1.0, value))
    min_freq = 110.0
    max_freq = 880.0
    return min_freq * (max_freq / min_freq) ** value

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  cap = cv2.VideoCapture(0)
  running = True

  current_tone = None

  while cap.isOpened() and running:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image,1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    ball_color = (0, 0, 0)
    
    if detection_result is not None:
      image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      
      #image = draw_bb_with_letter(image,detection_result,'A')
      if len(detection_result.hand_landmarks) > 0:
        landmarks = detection_result.hand_landmarks[0]
        # Obtener coordenadas del punto 8 (índice)
        index_finger_tip = landmarks[8]
                
        # Convertir coordenadas normalizadas a la pantalla de pygame
        screen_x = int(index_finger_tip.x * 640)
        screen_y = int(index_finger_tip.y * 480)

        lerped_y_value = screen_y / 480
        lerped_x_value = screen_x / 640

        new_color_x = (lerped_x_value) * 255
        new_color_y = (lerped_y_value) * 255

        
        note = value_to_note_frequency(lerped_y_value)

        if current_tone is not None:
          current_tone.stop()

        current_tone = play_tone(note)

        ball_color = (math.fabs(new_color_x), math.fabs(new_color_y), 0)

        # Actualizar posición del objeto en Pymunk
        body.position = screen_x, screen_y
        
    # Avanzar la simulación de Pymunk
    space.step(1 / 60.0)
    # Renderizar el objeto en Pygame
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, ball_color, (int(body.position.x), int(body.position.y)), int(circle.radius))
    
    pygame.display.flip()
    clock.tick(420)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
pygame.quit()
  