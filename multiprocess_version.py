# Imports
import matplotlib.pyplot as plt
import sys
import os
import io
import time
import random
import multiprocessing
from multiprocessing import Queue
import queue
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

    print("✅ Chrome and Chromedriver found. Environment ready.")


# Since we are using the chrome and chromedriver for testing,
# we have to set custom paths for selenium to find these.
options = Options()
# options.add_argument("--headless")
options.add_argument("--mute-audio")
options.binary_location = chrome_path

# Adding argument to load the browser at a certain resolution
options.add_argument("--window-size=800,600")

service = Service(driver_path)


# Set screen sizes for 4 sides in the screen
def set_screen_size(instance_id, screen_width, screen_height):
    if instance_id == 0:
        x = 0
        y = 0
    elif instance_id == 1:
        x = screen_width // 2
        y = 0
    elif instance_id == 2:
        x = 0
        y = screen_height // 2
    elif instance_id == 3:
        x = screen_width // 2
        y = screen_height // 2
    return x, y


# We need to disable automation flags so that selenium can load chrome Dino.


# Make the dino take an action
def send_keypress(action, driver):
    body = driver.find_element(By.TAG_NAME, "body")
    actions = ActionChains(driver)
    # Action 1 is jump, Action 0 is nothing for now
    if action == 1:
        body.send_keys(Keys.ARROW_UP)
    elif action == 2:
        #     # Duck
        actions.key_down(Keys.ARROW_DOWN).perform()
        time.sleep(0.2)
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
EPISODES = 250
# Let's also add the steps so that the agent doesn't get stuck
MAX_STEPS = 1000

# EXPLORE = 5000
# epsilon_decay = (epsilon - FINAL_EPSILON) / EXPLORE
epsilon_decay = 0.85
# Getting the batch size to train the network
# Checking if the batch size is enough to decide
BATCH_SIZE = 64
# Initializing the replay buffer
replay_buffer = ReplayBuffer(25000)

# Initialize the model
model = build_q_network()
target_model = build_q_network()
target_model.set_weights(model.get_weights())

# Declare the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


reward_history = []


# Parallel processing
class ChromeProcess(multiprocessing.Process):
    def __init__(
        self,
        instance_id,
        screen_width,
        screen_height,
        replay_experience_queue,
        epsilon_queue,
    ):
        super(ChromeProcess, self).__init__()
        self.instance_id = instance_id
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.replay_experience_queue = replay_experience_queue
        self.epsilon_queue = epsilon_queue

    def run(self):
        time.sleep(1)
        options = Options()

        # options.add_argument("--headless")
        options.add_argument("--mute-audio")
        options.binary_location = chrome_path
        # Adding argument to load the browser at a certain resolution
        options.add_argument("--window-size=800,600")
        service = Service(driver_path)
        epsilon = 0.66
        for episode in range(EPISODES):
            episode_reward = 0
            driver = webdriver.Chrome(service=service, options=options)
            try:
                # Set the window position of the browser
                x, y = set_screen_size(
                    self.instance_id, self.screen_width, self.screen_height
                )
                driver.set_window_position(x, y)
                driver.get("chrome://dino")
            except WebDriverException as e:
                if "net::ERR_INTERNET_DISCONNECTED" in str(e):
                    print(
                        f"⚠️  Instance {self.instance_id}: Expected offline Dino game loaded."
                    )
                else:
                    raise
            time.sleep(1)
            # Get frame
            frame = get_frame(driver)
            stacker = FrameStacker()
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
                # TODO epsilon
                # First, let's press space to play the game.
                # Choose action (0 = nothing, 1 = jump, 2 = duck)
                try:
                    new_epsilon = self.epsilon_queue.get_nowait()
                    epsilon = new_epsilon
                except queue.Empty:
                    # If no updated epsilon is available, use previous
                    pass
                if np.random.rand() < epsilon:
                    action = random.choice([0, 1, 2])  # Explore
                else:
                    print("PREDICTINGDASFADSFADSF")
                    # Creating a new variable here so that state dimension
                    # remains same for replay buffer
                    state_input = np.expand_dims(state, axis=0)
                    # print(state.shape)
                    # sys.exit()
                    q_values = model.predict(state_input)
                    print(q_values)
                    # q_values = model.predict(state)
                    action = np.argmax(q_values)
                    # if step % 10 == 0:
                    #     print(
                    #         f"[EP {episode+1}] Step {step} | Epsilon: {epsilon:.4f} | Q-values: {q_values[0]}"
                    #     )

                # Perform action using Selenium key events
                send_keypress(action, driver)

                #  Wait a bit, take next frame
                time.sleep(0.1)
                next_frame = get_frame(driver)
                if next_frame is None:
                    print("❌ Skipping frame due to capture issue.")
                    continue
                stacker.append(next_frame)

                # Award the agent with a reward if it survives
                reward = 1
                # Check if game is over
                done = check_game_over(driver)
                # If game over, decrease reward
                if done:
                    reward -= 20

                step += 1
                episode_reward += reward

                # Let's store the step in the replay buffer
                next_state = stacker.get_stacked_state()

                # Store experience in the queue
                self.replay_experience_queue.put(
                    (state, action, reward, next_state, done)
                )

                print(f"🎯 Episode {episode+1} reward: {episode_reward}")
                reward_history.append(episode_reward)

            driver.quit()

            print(f"✅ Browser {self.instance_id} closed.")


# Creating a models directory before saving
os.makedirs("models", exist_ok=True)
model.save("models/dino_q_network.keras")

# Run the script
if __name__ == "__main__":
    check_or_fail()
    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
    num_instances = 1
    # Creating a processes list to track every process
    processes = []
    # Implement queue because we want to share the replay buffer among the processes
    replay_experience_queue = Queue()
    # Queue for epsilon
    epsilon_queue = Queue(maxsize=1)
    epsilon_queue.put(0.99)
    # Setting up the environment
    # Making sure that the browser is centered
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    for i in range(num_instances):
        p = ChromeProcess(
            i, screen_width, screen_height, replay_experience_queue, epsilon_queue
        )
        p.start()
        processes.append(p)
        print("✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️✍️")

    training_step = 0
    while True:
        while not replay_experience_queue.empty():
            # Get the experience from the queue
            # Since this is a queue, we get the first one
            experience = replay_experience_queue.get()
            state, action, reward, next_state, done = experience
            replay_buffer.add(state, action, reward, next_state, done)

        # Train if enough samples
        if len(replay_buffer) >= 1000:
            batch = replay_buffer.sample(BATCH_SIZE)
            loss = train_step(model, target_model, batch, optimizer)
            print(f"Loss: {loss}")
            training_step += 1

            # Update epsilon here
            epsilon = epsilon_queue.get()
            epsilon *= epsilon_decay
            epsilon = max(0.1, epsilon)
            epsilon_queue.put(epsilon)
            # Update target network every 500 steps
            if training_step % 500 == 0:
                target_model.set_weights(model.get_weights())
                print("🔄 Target network updated.")
            # target_model.set_weights(model.get_weights())

        time.sleep(0.01)
    for p in processes:
        p.join()
    # print("Hello")
    # Collecting the replay experiences and training
