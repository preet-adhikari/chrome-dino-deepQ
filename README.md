# chrome-dino-deepQ
This is a project where the AI learns how to play the chrome dino game using Deep Q learning.

##### Issue while installing

## 🛡 Gatekeeper on macOS

I am running mac-arm64 for this project so both chrome and chromedriver are of the same version.

macOS may block `chromedriver` or the downloaded version of Chrome due to security checks.

To allow them to run:

```bash
xattr -d com.apple.quarantine ./bin/chromedriver/chromedriver-mac-arm64/chromedriver
xattr -d -r com.apple.quarantine "./bin/chrome/chrome-mac-arm64/Google Chrome for Testing.app"
```