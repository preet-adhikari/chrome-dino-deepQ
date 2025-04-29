##### Notes

The requirements are installed.

Selenium itself cannot open chrome://dino because it is a Chrome internal page. They are rendered by Chrome's UI layer and not the normal rendering engine.

Where selenium interacts with sites, this is it trying to interact with Chrome's UI.

Selenium throws a `net::ERR_INTERNET_DISCONNECTED` error because since dino game is triggered when chrome thinks you're offline, chrome://dino intentionally simulates a network disconnect to activate the offline page with the dino.

But once Selenium finds out that the network is disconnected, it triggers an error.

So catching the exception can help and we can ignore this error.

There are two ways to go about this. Either use JavaScript runtime to extract the states and then use Deep Q learning or use Visual based learning to learn the game by making the agent see and then go further.


I am going with the visual based learning option. 

To do that I am going to be taking screenshots. I wonder how would I store those and in such a way that it doesn't take up much space.

We can view images using Pillow, which is a Python Imaging Library.
 
We can capture the image using selenium's screenshot library and use Pillow to convert it into a PNG object and load it. 


###### Crop co-ordinates

This can be an issue while using different screens of different resolutions. Hence, to dynamically set a certain resolution and then going from there is the best option.

We use : options.add_argument("--window-size=800,600")

Now, we also want to center the screen so that it looks clean. 

There is a package called tkinter. It allows you to center the screen for the game. 

Now the steps we can take is, 
1. Capture the environment's frame
   
   Steps: 
   Taking a screenshot of the current Chrome window.
   Cropping it to just the game canvas
   Preprocessing the image into a normalized format

2. Stack frames to create motion
   
   Steps:
   Stack 3-4 recent frames together
   It gives the agent a sense of motion. 
   This stacked tensor is the input to the neural network

3. Perform an action
   
   The agent:
   Chooses an action (jump/duck/no-op)
    Emulate that in the browser via keyboard input(space, down, etc)

4. Wait a bit, capture the next frame

5. Assign a reward

6. Store this in "replay" memory
7. Repeat until game is over
8. Train the neural network

These are the steps.

First, let's capture and process a single clean game frame.
Capturing is done by selenium. 

Now, we're gonna use the crop_and_resize script to do the following things. First, we're gonna crop the image using these values : (10, 25, 1200, 600) and then we're converting them into grayscale so that the images can be processed faster. 

We're resizing the image to (84,84) because it is the optimal resize and the DeepMind Atari DQN paper standardized on 84 x 84 grayscale frames.

Once we have the image, we need to convert that into a numpy array to feed it into the neural network. 


Here, we need to stack frames on top of each other. Capturing just a single frame will not give the entire context. Let's say the dino is in mid-air on a frame. The neural network doesn't recognize what the dino is doing or not. 

So we will stack frames on each other and we'll use 4 frames. 4 frames is enough to determine and help the neural network learn. 

The shape will be 84x84x4  of the data that will be fed into the neural network. 

For stacking frames, we'll use queue because of it's First in first out architecture that allows us to store and update frames accordingly.

We'll be using python's package deque. 

We'll create a class FrameStacker that takes in frames and adds frame to the queue. 

The FrameStacker class takes in the frames, and generates a 84x84x4 shape output for the neural network. 

Then we implement a function for the dino to take action. We use selenium to grab the body element and use its send_keys function to set 1 as jump(arrow up) key and 0 as duck(down key).

When the dino game starts, Chrome initiates a global Javascript object called Runner.instance_

When the dino crashes, this global variable's property is set to Runner.instance_.crashed = true. We can check that to check whether the game is over or not. 

So we can use selenium to execute a javascript script and check if the game is over inside the episode. 

As of right now, these are the steps we have:

We define the browser width, load the browser. We set the number of episodes. Then for each episode, we load the browser and dino game, and get the frame from the current input. The frame is multiplied four times for the first time and is stacked on top of each other. Then once the actions are taken, the frames are generated. After generating each frame they can be stacked on top of each other. 

Once the action is taken, new frames are added until the dino doesn't die. We count how many steps until the dino runs. This was the result of the first run: 

```                                                                
✅ Chrome and Chromedriver found. Environment ready.
Starting episode: 1: 
⚠️  Expected error: Dino game loaded offline.
Episode ended after 30 steps.
Press Enter to close the browser...
```

Next, let's build the reward system for training the DQN.

This is the reinforcement part of learning. We will implement a system where survival will be the reward. So for each step the dino survives, reward will be added by 1. If it dies however, it will be subtracted by 100. 

Surviving encourages the agent to stay alive longer where dying and getting reward cut off like that, will make the agent not take bad actions. 

For the DeepQ learning, we're going to use the replay buffer. 

This Replay buffer stores in the replays so that it uses the DeepQ learning paradigm. After getting the steps from the replay, we store the experience in a buffer which is a queue. In this buffer, we initialize a queue, and then we store state, action, reward, next_state and done variables. Then, from this replay buffer, we can generate a random sample which can be used by the network to extract previous knowledge so that it can learn and develop faster. 

We initialize the replay buffer before we start the episodes because we want the buffer to persist across all the episodes.

Now we have stored the experience. Once we store all the experiences, we need to check if all the experiences stored exceed a certain batch size. 

While creating the replay buffer, we initialize a cpacity for the buffer. This is the buffer memory, which specifies how many steps of the game does the buffer store. Then we have the batch size, which will be dividing that memory into batches and then train the model on those batches. 

We'll set a random buffer size of 10,000.

Then, when the replay_buffer is greater than the batch size, it means the model can train on that data. So, in that case, we will unzip the data into the states, actions, rewards and next_states plus dones that can be taken and put into the Deep Q learning network. 


Tricky part. First we were only considering to implement jump and do nothing for each step. Now, we will be adding 'duck' functionality as well. We will be implementing '2' as the action for duck. But for duck action, it can get a little complicated because we have to hold down the duck button. For that, we used selenium actions.

Now, finally we are ready to create the neural network to train this data. We will be using a CNN layer, a Flatten layer and a Dense layer to generate the outputs of the Q network. We need to feed the target to this Q network which will then output a number of actions to take. 

And for Q network, we need to use gradient tape because we're calculating the loss manually and hence tensorflow needs a way to compute the gradients. Hence, we use GradientTape() because through that, the model can calculate and use the gradients properly. The gradient tape tracks all the gradients in the Q network and you can formulate it to teach the network properly.

We use huber loss instead of mean squared error because huber loss smooths out large errors and focuses on making, smaller, stable improvements.

We used the simple function of crop and resize to get the frame, but since the model wasn't doing well, extracting the image properly is the next step first. 

We'll be using the javascript canvas to get the area and then use screenshot to get the image. 

Canvas details directly retrieves the position and size of game canvas. 

'''
{'bottom': 185, 'height': 150, 'left': 124, 'right': 676, 'toJSON': {}, 'top': 35, 'width': 552, 'x': 124, 'y': 35}
'''

Actual width of the canvas in canvas pixels, which is 1104. This can help to maintain aspect ratio or consistency in resolution. We're taking dino width when it comes to ducking so that detection can be done better.

We're using skimage, which is the scikit image for image processing. It has many features. 

A better idea would be to look at the screen ahead of the dino, rather than what's behind, hence we used the dino width for ducking because that is the most of what we can get of the screen. 

This time, we'll be setting a size of 75x75 since that seems better to work with. 

This model needed tweaking: 

''' 
def build_q_network(input_shape=(84, 84, 4), num_actions=2):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=8, strides=4,
                      activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=4, strides=2,
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=3, strides=1,
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions)  # no activation (raw Q-values)
    ])
    return model
'''

We went with this one: 
'''
 model = models.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), strides=2,
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), strides=1,
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')  
    ])

    # Compile and return model
    model.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
'''

We encountered an error while training the model. The size that was being generated from training added another dimension.
Something like:
'''
State 1 shape: (1, 75, 75, 3)
State 2 shape: (1, 75, 75, 3)
State 3 shape: (1, 75, 75, 3)
'''

The issue could be in the replay buffer.

The issue was that the replay buffer was storing an extra dimension while storing the state. The extra dimension was originally added as a batch size so when the state is passed into the model.predict function, the extra dimension is the batch size. 

But the replay buffer only needs frames of the same shape and when it unzips to train, it throws an error of a mismatching shape. 

Fixing the shape issue was easy, as I didn't store the expanded dimensions of the frame into the replay buffer. 

After that, it started running smoothly. The model is still not learning however. 

I have decided to go with parallel processing, with a distributed experience collection in mind. What that is is I will be loading 4 different browser windows on one global frame buffer to traing one agent. This way, I have 4 workers for one model which means training and testing will be much much faster. 

There is a complication, however. Its where multiprocessing doesn't allow variables like list to be shared across processes. This can be an issue because we're storing variables into a single buffer. 

In multiprocessing in Python, when you create a new process, it forks(copies), the parent process into a new seperate memory space, and then each process becomes independent and each process has its own copy of variables.

Hence a separate queue or list is necessary to share the variables between processes because we'll be storing frames and we need to share the frames between each other. 

Where in multithreading you share memory, in multiprocessing, it is not the case. Hence, we use queue to share the variables between processes. Using that, we can then share the replay buffer, which is the frames across the 4 different browsers we use. 

Now, the training of the model only happens in the main process. This way, it is much much easier. 

There was an issue while using multiprocessing. The epsilon value was not being tracked well in the implementation. What I mean is I got stuck trying to figure out the best way to track the epsilon value because I needed to implement a global epsilon value due to the limitations of multiprocessing. If I use a local variable, it shrinks too much, the 4 workers are kind of working on their own. I tried adding it to the training set but I couldn't find the perfect balance of ending an episode and then training the model. It seemed like it would take a little more time than I thought finding the perfect balance and it is an optimization step, because my first issue is that our model isn't learning at all. And figuring out a simple epsilon method would be much much easier to implement. 

Hence, I am going back to the single process run. I have kept the code of multiprocessing in this repo and it will be something I can come back later and try to figure out. 


For now, back to the model. I have also have a hunch that using softmax as my activation function is not the right way to go because this isn't a classification problem, this is a reward maximization problem. 

Hence, I am removing softmax. 

Training with 
'''
epsilon = 0.99
epsilon_decay = 0.993
'''

The model is still not learning.

Finally, the model seems to learn. After increasing the frame stack to 4 and changing the input to 84x84, the model seemed to learn. While it still can't get past the 164 score, which was its best with the three 2000 episodes tests I did, there are a lot of improvements to be made. One of them was the input lag on how it was taking the frames. While a little time sleep is necessary to catch the next frame, it still seems like there was lag in this input. And since the version of chrome dino is a little faster, maybe training this on a slower version would have been better. Also, the reward system. After shifting the death penalty to -5, it seemed to do better. Maybe decreasing this reward would help as well. One of the other issues was actually the crop mechanism wasn't working right. Seems like the canvas pixels and the screenshot pixels weren't matching at all and hence we had to divert to a hardcoded approach of cropping out the frames.

Finally, maybe selenium is not the best approach for this game. A more robust mechanism would have been much better, or maybe the model wasn't deep enough for it. Fine tuning hyperparameters for better performance is the way to go. The frame preprocessing too, I think that can also be looked at and improved upon. 

This was a fun project. We learned a lot of things. Even in some failed setups, we did manage to generate something that views and learns. 





