import cv2
import time
import math
import pygame
from hand_tracking import HandLandmarker, HandLandmarkerOptions, BaseOptions, VisionRunningMode, get_result, draw_landmarks_on_image
from audio import play_tone, value_to_note_frequency
from physics import create_circle_body, setup_space
import mediapipe as mp

model_path = 'hand_landmarker.task'

def main():

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()

    space = setup_space()
    body, circle = create_circle_body()
    space.add(body, circle)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=get_result,
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        running = True
        current_tone = None

        while cap.isOpened() and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image,1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            ball_color = (0,0,0)
            from hand_tracking import detection_result
            if detection_result is not None:
                image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                if len(detection_result.hand_landmarks) > 0:
                    landmarks = detection_result.hand_landmarks[0]
                    index_tip = landmarks[8]
                    screen_x = int(index_tip.x * 640)
                    screen_y = int(index_tip.y * 480)
                    lerped_y_value = screen_y / 480
                    lerped_x_value = screen_x / 640
                    new_color_x = lerped_x_value * 255
                    new_color_y = lerped_y_value * 255
                    note = value_to_note_frequency(lerped_y_value)
                    if current_tone is not None:
                        current_tone.stop()
                    current_tone = play_tone(note)
                    ball_color = (math.fabs(new_color_x), math.fabs(new_color_y), 0)
                    body.position = screen_x, screen_y

            space.step(1/60)
            screen.fill((255,255,255))
            pygame.draw.circle(screen, ball_color, (int(body.position.x), int(body.position.y)), int(circle.radius))
            pygame.display.flip()
            clock.tick(420)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


main()