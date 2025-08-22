# SECRET_KEY Configuration Fix

## Issue
The bot is failing to start because the SECRET_KEY environment variable is not properly configured on the server.

## Solution
You need to set the SECRET_KEY environment variable on your server with this value:
```
SECRET_KEY=0f764b733691e20754e00e6c88c039d3c531fd773b22c521a37e9e5817f0373c
```

## How to Set on Different Platforms

### Railway
Add this to your Railway environment variables:
```
SECRET_KEY=0f764b733691e20754e00e6c88c039d3c531fd773b22c521a37e9e5817f0373c
```

### Docker
If using docker-compose, add to your .env file or docker-compose.yml:
```yaml
environment:
  - SECRET_KEY=0f764b733691e20754e00e6c88c039d3c531fd773b22c521a37e9e5817f0373c
```

### Direct Server (Linux/Ubuntu)
Export the environment variable:
```bash
export SECRET_KEY=0f764b733691e20754e00e6c88c039d3c531fd773b22c521a37e9e5817f0373c
```

Or add to your systemd service file if using systemd.

### Heroku
```bash
heroku config:set SECRET_KEY=0f764b733691e20754e00e6c88c039d3c531fd773b22c521a37e9e5817f0373c
```

## Important Notes
- This SECRET_KEY is used for encryption and must be at least 32 characters
- Keep this value secure and never share it publicly
- The same SECRET_KEY must be used consistently across deployments
- After setting the environment variable, restart your application

## Verification
After setting the SECRET_KEY, the validation errors should be resolved and the bot should start successfully.