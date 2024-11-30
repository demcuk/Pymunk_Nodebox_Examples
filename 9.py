#Переіменовані деякі функції щоб зробити їх більш зрозумілими create_body, move_robot_towards_target, move_robot_with_q_learning.
#Замість strategy і strategy2 тепер використовується move_robot_towards_target та move_robot_with_q_learning для чіткої структуризації.
#Оновлено функцію simulate_friction для плавного зниження швидкості.
#Перероблено вибір дії робота для покращення стабільності стратегії з використанням epsilon-greedy алгоритму.

from __future__ import division
from nodebox.graphics import *
import pymunk
import pymunk.pyglet_util
import random, math
import numpy as np

space = pymunk.Space()

def create_body(x, y, shape, *shape_args):
    body = pymunk.Body()
    body.position = x, y
    shape_obj = shape(body, *shape_args)
    shape_obj.mass = 1
    shape_obj.friction = 1
    space.add(body, shape_obj)
    return shape_obj

# Основні об'єкти
robot_1 = create_body(300, 300, pymunk.Poly, ((-20, -5), (-20, 5), (20, 15), (20, -15)))
robot_1.score = 0
robot_2 = create_body(200, 300, pymunk.Poly, ((-20, -5), (-20, 5), (20, 15), (20, -15)))
robot_2.color = (0, 255, 0, 255)
robot_2.score = 0
robot_2.body.q_values = [[0, 0], [0, 0], [0, 0]]  # Q-таблиця для навчання
robot_2.body.action = 0  # 0 - залишити напрямок, 1 - змінити

target = create_body(300, 200, pymunk.Circle, 10, (0, 0))  # Ціль
obstacles = [create_body(350, 250, pymunk.Circle, 10, (0, 0))]  # Перешкоди

def calculate_angle(x, y, x1, y1):
    return math.atan2(y1 - y, x1 - x)

def calculate_distance(x, y, x1, y1):
    return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

def is_in_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

def is_in_sector(x, y, cx, cy, r, angle):
    robot_angle = calculate_angle(cx, cy, x, y) % (2 * math.pi)
    angle = angle % (2 * math.pi)
    return is_in_circle(x, y, cx, cy, r) and (angle - 0.5 < robot_angle < angle + 0.5)

# Стратегія руху для робота (без випадкових виборів)
def move_robot_towards_target(robot_body):
    speed = 100
    angle = robot_body.angle
    robot_body.velocity = speed * math.cos(angle), speed * math.sin(angle)
    x, y = robot_body.position
    distance_to_target = calculate_distance(x, y, 350, 250)
    if distance_to_target > 180:  # Якщо робот далеко від центра
        robot_body.angle = calculate_angle(x, y, 350, 250)  # Рухатися в бік центра

# Стратегія з Q-навчанням для робота
def move_robot_with_q_learning(robot_body):
    speed = 100
    angle = robot_body.angle
    robot_body.velocity = speed * math.cos(angle), speed * math.sin(angle)
    x, y = robot_body.position
    distance_to_target = calculate_distance(x, y, 350, 250)

    if canvas.frame % 10 == 0:  # Оновлення кожні 10 кадрів
        is_target_in_sector = is_in_sector(target.body.position[0], target.body.position[1], x, y, 100, angle)
        is_obstacle_in_sector = is_in_sector(obstacles[0].body.position[0], obstacles[0].body.position[1], x, y, 100, angle)

        # Визначаємо стан та винагороду
        if is_target_in_sector:
            state = 1
            reward = 1 if robot_body.action == 0 else -1
        elif is_obstacle_in_sector:
            state = 2
            reward = -1 if robot_body.action == 0 else 1
        else:
            state = 0
            reward = 0

        # Оновлюємо Q-таблицю
        robot_body.q_values[state][robot_body.action] += reward

        # Стратегії вибору дій
        q_value = robot_body.q_values[state][robot_body.action]
        if random.random() < abs(1.0 / (q_value + 0.1)):  # epsilon-greedy
            robot_body.action = random.choice([0, 1])  # випадкове вибрання дії
        else:
            robot_body.action = np.argmax(robot_body.q_values[state])  # вибір найкращої дії

        # Якщо вибрано змінити напрямок
        if robot_body.action == 1:
            robot_body.angle = 2 * math.pi * random.random()

    # Перешкоди та запобігання виїзду з кола
    if distance_to_target > 180:
        robot_body.angle = calculate_angle(x, y, 350, 250)

# Функція для оновлення рахунку
def update_score(robot, target_robot_1, target_robot_2, p=1):
    bx, by = robot.body.position
    target_robot_1_x, target_robot_1_y = target_robot_1.body.position
    target_robot_2_x, target_robot_2_y = target_robot_2.body.position
    if not is_in_circle(bx, by, 350, 250, 180):  # Якщо робот за межами кола
        if calculate_distance(bx, by, target_robot_1_x, target_robot_1_y) < calculate_distance(bx, by, target_robot_2_x, target_robot_2_y):
            target_robot_1.score += p  # Оновлення рахунку
        else:
            target_robot_2.score += p
        robot.body.position = random.randint(200, 400), random.randint(200, 300)  # Позиціонування робота випадково

def update_game_score():
    update_score(target, robot_1, robot_2)
    for robot in obstacles:
        update_score(robot, robot_1, robot_2, p=-1)

# Управління роботом вручну (через клавіатуру та мишу)
def manual_control():
    speed = 10
    robot = robot_1.body
    angle = robot.angle
    x, y = robot.position
    vx, vy = robot.velocity
    if canvas.keys.char == "a":
        robot.angle -= 0.1
    if canvas.keys.char == "d":
        robot.angle += 0.1
    if canvas.keys.char == "w":
        robot.velocity = vx + speed * math.cos(angle), vy + speed * math.sin(angle)
    if canvas.mouse.button == LEFT:
        robot.angle = calculate_angle(x, y, *canvas.mouse.xy)
        robot.velocity = vx + speed * math.cos(angle), vy + speed * math.sin(angle)

# Функція для симуляції фрикції
def simulate_friction():
    for robot in [robot_1, target, robot_2] + obstacles:
        robot.body.velocity = robot.body.velocity[0] * 0.9, robot.body.velocity[1] * 0.9  # Зниження швидкості
        robot.body.angular_velocity *= 0.9  # Зниження обертальної швидкості

# Основна функція для малювання та обробки кадрів
draw_options = pymunk.pyglet_util.DrawOptions()

def draw(canvas):
    canvas.clear()
    fill(0, 0, 0, 1)
    text(f"{robot_1.score} {robot_2.score}", 20, 20)
    nofill()
    ellipse(350, 250, 350, 350, stroke=Color(0))
    manual_control()
    move_robot_with_q_learning(robot_2.body)
    update_game_score()
    simulate_friction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)
