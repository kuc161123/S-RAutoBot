import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

print(f"CWD: {os.getcwd()}")
print(f"Sys Path: {sys.path}")

try:
    print("Attempting to import autobot.core.bot...")
    import autobot.core.bot
    print("Successfully imported autobot.core.bot")
except ImportError as e:
    print(f"Failed to import autobot.core.bot: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error importing autobot.core.bot: {e}")
    # Continue to check telegram even if bot fails, to see more errors
    # sys.exit(1)

try:
    print("Attempting to import autobot.core.telegram...")
    import autobot.core.telegram
    print("Successfully imported autobot.core.telegram")
except ImportError as e:
    print(f"Failed to import autobot.core.telegram: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error importing autobot.core.telegram: {e}")
    sys.exit(1)

print("Verification successful!")
