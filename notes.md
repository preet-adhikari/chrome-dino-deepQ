##### Notes

The requirements are installed.

Selenium itself cannot open chrome://dino because it is a Chrome internal page. They are rendered by Chrome's UI layer and not the normal rendering engine.

Where selenium interacts with sites, this is it trying to interact with Chrome's UI.

Selenium throws a `net::ERR_INTERNET_DISCONNECTED` error because since dino game is triggered when chrome thinks you're offline, chrome://dino intentionally simulates a network disconnect to activate the offline page with the dino.

But once Selenium finds out that the network is disconnected, it triggers an error.

So catching the exception can help and we can ignore this error.

There are two ways to go about this. Either use JavaScript runtime to extract the states and then use Deep Q learning or use Visual based learning to learn the game by making the agent see and then go further.
