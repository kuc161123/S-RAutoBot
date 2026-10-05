"""Source-controlled release decision; Telegram and models cannot override it."""

LIVE_APPROVED = False
REASON = (
    "Live release blocked: the Apex numeric interpretation has no demonstrated "
    "trading edge. Representative acceptance replays have too few completed "
    "trades to validate the strategy. Shadow/testnet validation and strategy "
    "review are required."
)
