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
8. Train your neural network

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








