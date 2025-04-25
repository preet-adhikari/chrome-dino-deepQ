# demo.py
import tkinter as tk
import sys
import os
import io
import time
from PIL import Image
import numpy as np
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from framestacker import FrameStacker
from crop_and_resize import crop_and_resize
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/dino_q_network.h5")

# Chrome setup
chrome_path = "./bin/chrome/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
driver_path = "./bin/chromedriver/chromedriver-mac-arm64/chromedriver"

options = Options()
options.binary_location = chrome_path
options.add_argument("--window-size=800,600")
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)

# Center screen
root = tk.Tk()
x = (root.winfo_screenwidth() - 800) // 2
y = (root.winfo_screenheight() - 600) // 2
driver.set_window_position(x, y)


def send_keypress(action, driver):
    body = driver.find_element(By.TAG_NAME, "body")
    actions = ActionChains(driver)
    if action == 1:
        body.send_keys(Keys.ARROW_UP)
    elif action == 2:
        actions.key_down(Keys.ARROW_DOWN).perform()
        time.sleep(0.1)
        actions.key_up(Keys.ARROW_DOWN).perform()


def check_game_over(driver):
    return driver.execute_script("return Runner.instance_.crashed")


def get_frame():
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    image = crop_and_resize(image)
    frame = np.array(image).astype(np.float32) / 255.0
    return frame


# Load game
try:
    driver.get("chrome://dino")
except WebDriverException as e:
        if "net::ERR_INTERNET_DISCONNECTED" in str(e):
            print("⚠️  Expected error: Dino game loaded offline.")
        else:
            raise
time.sleep(1)
frame = get_frame()
stacker = FrameStacker()
stacker.reset(frame)
driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)

done = False
step = 0

while not done:
    state = stacker.get_stacked_state()
    q_values = model(np.expand_dims(state, axis=0), training=False)
    action = np.argmax(q_values[0])
    send_keypress(action, driver)
    time.sleep(0.1)
    frame = get_frame()
    stacker.append(frame)
    step += 1
    done = check_game_over(driver)
    print(f"Demo step: {step}, action: {action}")

print("🦖 Game Over")
driver.quit()
