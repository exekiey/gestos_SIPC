import pymunk

def create_circle_body(position=(320, 240), radius=20):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position
    circle = pymunk.Circle(body, radius)
    return body, circle

def setup_space():
    space = pymunk.Space()
    space.gravity = (0, 0)
    return space
