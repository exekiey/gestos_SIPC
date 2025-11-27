import pygame
import numpy as np

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def play_tone(freq: float):
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
