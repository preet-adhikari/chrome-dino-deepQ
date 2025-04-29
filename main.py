# Imports
import matplotlib.pyplot as plt
import sys
import os
import io
import time
import random
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import numpy as np
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import tensorflow as tf
import skimage as skimage
from skimage import transform, color, exposure, io
from skimage.transform import rotate
from skimage.util import img_as_uint
from replay_buffer import ReplayBuffer
from framestacker import FrameStacker


# Import Q Network
from q_network import build_q_network, train_step

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

    print("Chrome and Chromedriver found. Environment ready ✅.")


check_or_fail()
# Since we are using the chrome and chromedriver for testing,
# we have to set custom paths for selenium to find these.
options = Options()
# options.add_argument("--headless")
options.add_argument("--mute-audio")
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
    actions = ActionChains(driver)
    # Action 1 is jump, Action 0 is nothing for now
    if action == 1:
        body.send_keys(Keys.ARROW_UP)
    elif action == 2:
        # Duck
        actions.key_down(Keys.ARROW_DOWN).perform()
        time.sleep(0.1)
        actions.key_up(Keys.ARROW_DOWN).perform()
    else:  # Do nothing
        pass


# Check if the game is over or not
def check_game_over(driver):
    return driver.execute_script("return Runner.instance_.crashed")


# Get frame from the browser window
def get_frame(driver):
    canvas_details = driver.execute_script(
        "return Runner.instance_.canvas.getBoundingClientRect()"
    )
    actual_width = driver.execute_script("return Runner.instance_.canvas.width")
    dino_width = driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK")
    xPos = driver.execute_script("return Runner.instance_.tRex.xPos")
    try:
        screenshot = driver.get_screenshot_as_png()

        # Crop and resize frame
        frame = crop_and_resize(
            screenshot, canvas_details, actual_width, dino_width, xPos
        )
        return frame
    except Exception as e:
        print(f"❌ get_frame failed: {e}")
        return None


# Let's set the number of episodes first
EPISODES = 2000

# Let's also add the steps so that the agent doesn't get stuck
MAX_STEPS = 1000


epsilon = 1.0
epsilon_decay = 0.998
epsilon_min = 0.10

BATCH_SIZE = 64

# Initializing the replay buffer
replay_buffer = ReplayBuffer(25000)

# Initialize the model
# model = build_q_network()
# Trying the 84x84 version
model = build_q_network(input_shape=(84, 84, 4))
# target_model = build_q_network()
target_model = build_q_network(input_shape=(84, 84, 4))
target_model.set_weights(model.get_weights())

# Declare the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

global_step = 0
reward_history = []
previous_score = 0
for episode in range(EPISODES):
    print(f"Starting episode: {episode + 1}: ")
    episode_reward = 0
    # Open
    try:
        driver.get("chrome://dino")
    except WebDriverException as e:
        if "net::ERR_INTERNET_DISCONNECTED" in str(e):
            print("⚠️  Expected error: Dino game loaded offline.")
        else:
            raise
    time.sleep(1)
    # Execute in 2x speed
    # driver.execute_script("Runner.instance_.setSpeed(2)")
    # Let's take the first frame
    frame = get_frame(driver)
    # stacker = FrameStacker()
    # For 84x84
    stacker = FrameStacker(frame_shape=(84, 84))
    if frame is None:
        print("❌ Initial frame is None. Skipping episode.")
        continue
    stacker.reset(frame)

    # Start the game by pressing space once
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
    done = False
    step = 0
    reward = 0

    while not done and step < MAX_STEPS:
        # Get state from stacker
        state = stacker.get_stacked_state()

        # First, let's press space to play the game.
        # Choose action (0 = nothing, 1 = jump, 2 = duck)
        if np.random.rand() < epsilon:
            action = random.choice([0, 1, 2])  # Explore
        else:
            # Creating a new variable here so that state dimension
            # remains same for replay buffer
            state_input = np.expand_dims(state, axis=0)
            # print(state.shape)
            # sys.exit()
            q_values = model.predict(state_input)
            print(q_values)
            # q_values = model.predict(state)
            action = np.argmax(q_values)

        # Perform action using Selenium key events
        send_keypress(action, driver)

        #  Wait a bit, take next frame
        # time.sleep(0.1)
        # Changed time sleep here
        time.sleep(0.04)
        # Get score
        try:
            current_score = int(
                driver.execute_script(
                    "return Runner.instance_.distanceMeter.digits.join('')"
                )
            )
            score_diff = current_score - previous_score

            if score_diff > 0:
                reward = score_diff * 0.1
            else:
                reward = 0.1

            previous_score = current_score
        except:
            reward = 0.1

        done = check_game_over(driver)
        # Game over penalty
        if done:
            reward = -5
        next_frame = get_frame(driver)
        if next_frame is None:
            print("Skipping frame due to capture issue.")
            continue
        stacker.append(next_frame)

        step += 1
        global_step += step
        episode_reward += reward

        # Let's store the step in the replay buffer
        next_state = stacker.get_stacked_state()
        replay_buffer.add(state, action, reward, next_state, done)

        # We take a sample of the batch size.
        if len(replay_buffer) >= 1000:
            batch = replay_buffer.sample(BATCH_SIZE)
            # Initialize the model
            loss = train_step(model, target_model, batch, optimizer)
            print(f"Loss: {loss}")

        print(f"Step: {step}, Reward: {reward}", end="\n")
        if global_step % 100 == 0:
            target_model.set_weights(model.get_weights())
        # print("🔄 Updated target network.")
    print(f"🎯 Episode {episode+1} reward: {episode_reward}")
    reward_history.append(episode_reward)
    print(f"🎯 Total reward for episode {episode+1}: {episode_reward}")

    # Decay epsilon

    epsilon *= epsilon_decay
    epsilon = max(epsilon_min, epsilon)
    # driver.quit()
    print(f"Episode ended after {step} steps.")


def get_unique_filename(filename):
    """
    Returns a unique filename by appending _1, _2, etc. if needed.
    """
    if not os.path.exists(filename):
        return filename  # File doesn't exist, use original

    base, ext = os.path.splitext(filename)
    counter = 1

    new_filename = f"{base}_{counter}{ext}"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"

    return new_filename


# Creating a models directory before saving
os.makedirs("models", exist_ok=True)
file_name = "models/dino_q_network.keras"
unique_filename = get_unique_filename(file_name)
print("Saving file as:", unique_filename)
model.save(unique_filename)

input("Press Enter to close the browser...")


driver.quit()

def moving_avg(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# Plot episode results
plt.figure(figsize=(10, 4))  
# Plot raw episode rewards
plt.plot(reward_history, label="Episode Reward")
# Plot moving average
if len(reward_history) >= 10:
    plt.plot(moving_avg(reward_history), label="Moving Avg (10)", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress of Dino Q-Learning Agent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

input("Press Enter to begin training:")


# Testing mode
print("\n🧪 Testing trained model (no exploration)...\n")
test_episodes = 2  # or however many you want

for test_ep in range(test_episodes):
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("chrome://dino")
    except WebDriverException as e:
        if "net::ERR_INTERNET_DISCONNECTED" in str(e):
            print("⚠️  Expected error: Dino game loaded offline.")
        else:
            raise
    time.sleep(1)
    frame = get_frame(driver)
    if frame is None:
        print("❌ Skipping test episode due to bad initial frame.")
        continue
    stacker.reset(frame)
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)

    done = False
    step = 0

    while not done:
        state = stacker.get_stacked_state()
        q_values = model(np.expand_dims(state, axis=0), training=False)
        action = np.argmax(q_values[0])  # Always exploit
        send_keypress(action, driver)
        time.sleep(0.1)
        next_frame = get_frame(driver)
        if next_frame is None:
            print("❌ Frame capture failed.")
            break
        stacker.append(next_frame)
        step += 1
        done = check_game_over(driver)
        print(f"[TEST] Step {step}, Action: {action}")
    driver.quit()
    print(f"✅ Test episode {test_ep+1} ended after {step} steps.")
