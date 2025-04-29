# chrome-dino-deepQ
This is a project where the AI learns how to play the chrome dino game using Deep Q learning.

##### Issue while installing

## Gatekeeper on macOS

I am running mac-arm64 for this project so both chrome and chromedriver are of the same version.

macOS may block `chromedriver` or the downloaded version of Chrome due to security checks.

To allow them to run:

```bash
xattr -d com.apple.quarantine ./bin/chromedriver/chromedriver-mac-arm64/chromedriver
xattr -d -r com.apple.quarantine "./bin/chrome/chrome-mac-arm64/Google Chrome for Testing.app"
```


## Create a virtual environment and run

```bash
python3 -m venv dino-env
source venv/bin/activate
```

## Running the code

Run the for training using
```bash
python main.py
```

The model will be saved on models directory.

## Running the demo after training

You can use this command to run the demo

```bash
python demo.py
```

Note: This model still struggles with learning well and playing well. There are improvements that can be made here. One of the things that can be done is stacking more frames and using a more real time screenshot reader like chrome devtools or better frame processing than the one is there. This can be improved upon, so any suggestions are appreciated. 
