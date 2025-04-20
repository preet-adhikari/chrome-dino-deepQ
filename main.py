# Imports
import sys
import os
import io
import time
import random
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from PIL import Image
import numpy as np
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from framestacker import FrameStacker

# To center screen
import tkinter as tk

# For crop and resize
from crop_and_resize import crop_and_resize

# Chrome and Chromedriver paths
chrome_path = "./bin/chrome/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
driver_path = "./bin/chromedriver/chromedriver-mac-arm64/chromedriver"

def check_or_fail():

    if not os.path.exists(chrome_path):
        print(f"❌ Chrome not found at: {chrome_path}")
        sys.exit(1)

    if not os.path.exists(driver_path):
        print(f"❌ Chromedriver not found at: {driver_path}")
        sys.exit(1)

    print("✅ Chrome and Chromedriver found. Environment ready.")


# Check if chrome and chromedriver are present
check_or_fail()

# Since we are using the chrome and chromedriver for testing, 
# we have to set custom paths for selenium to find these.
options = Options()
options.binary_location = chrome_path

# Adding argument to load the browser at a certain resolution
options.add_argument("--window-size=800,600")

service = Service(driver_path)


# Setting up the environment

# Making sure that the browser is centered
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 800) // 2
y = (screen_height - 600) // 2

# We need to disable automation flags so that selenium can load chrome Dino.

driver = webdriver.Chrome(service=service, options=options)

# Set the window position of the browser
driver.set_window_position(x, y)


# Make the dino take an action
def send_keypress(action, driver):
    body = driver.find_element(By.TAG_NAME, "body")
    # Action 1 is jump, Action 0 is nothing for now
    if action == 1: 
        body.send_keys(Keys.ARROW_UP)
    # elif action == 0:
        # body.send_keys(Keys.ARROW_DOWN)

# Check if the game is over or not
def check_game_over(driver):
    return driver.execute_script("return Runner.instance_.crashed")

# Get frame from the browser window
def get_frame():
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    image = crop_and_resize(image)
    frame = np.array(image).astype(np.float32) / 255.0
    return frame

# Let's set the number of episodes first
EPISODES = 1

for episode in range(EPISODES):
    print(f"Starting episode: {episode + 1}: ")
    # Open
    try:
        driver.get("chrome://dino")
    except WebDriverException as e:
        if "net::ERR_INTERNET_DISCONNECTED" in str(e):
            print("⚠️  Expected error: Dino game loaded offline.")
        else:
            raise
    time.sleep(1) # Wait for it to load
    
    # Let's take the first frame
    frame = get_frame()
    stacker = FrameStacker()
    stacker.reset(frame)

    # Start the game by pressing space once
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
    done = False
    step = 0

    while not done:
        # Get state from stacker
        state = stacker.get_stacked_state()

        # First, let's press space to play the game.
        # Choose action (right now: random or always jump)
        # TODO : define: 0 = do nothing, 1 = jump
        action = random.choice([0, 1])

        # TODO : Perform action using Selenium key events
        send_keypress(action, driver)  # TODO : implement this

        # 6. Wait a bit, take next frame
        time.sleep(0.1)  # or faster/slower based on game
        next_frame = get_frame()
        stacker.append(next_frame)

        # Check for collision 
        done = check_game_over(driver)  # TODO : implement this
        step += 1  # TODO : or use actual reward signal later
        print(f"Step: {step}", end="\r")
        
    print(f"Episode ended after {step} steps.")

input("Press Enter to close the browser...")

driver.quit()
