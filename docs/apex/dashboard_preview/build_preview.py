"""Offline synthetic Telegram layout preview; never initializes a network client.

Run from the repository root:
  PYTHONPATH=. python docs/apex/dashboard_preview/build_preview.py
"""

import asyncio
import json
import time
from pathlib import Path

from apex_bot.config import Config
from apex_bot.service import Service
from apex_bot.storage import Store
from apex_bot.telegram import TelegramController, VIEWS


async def build():
    now = time.time()
    store = Store(sqlite_path=":memory:")
    await store.initialize()
    await store.lease(ttl=10000)
    config = Config(
        universe_mode="static",
        symbols=("BTCUSDT", "ETHUSDT"),
        telegram_token="123456:SYNTHETIC_PREVIEW",
        telegram_chat_id="1",
        telegram_user_ids=frozenset({1}),
        openai_key="SYNTHETIC_ONLY",
        openai_model="fixture-model",
    )
    service = Service(config, store)
    controller = TelegramController(None, config.telegram_token, "1", {1}, service)
    fixtures = {}
    for i in range(14):
        op = {
            "id": str(i),
            "symbol": "BTCUSDT" if i % 2 else "ETHUSDT",
            "side": "Buy" if i % 2 else "Sell",
            "state": "WAIT",
            "setup": "2" if i % 2 else "2S",
            "entry": 100,
            "stop": 90 if i % 2 else 110,
            "invalidation": 92 if i % 2 else 108,
            "target1": 120 if i % 2 else 80,
            "target2": 140 if i % 2 else 60,
            "reason": "AWAIT_CLOSED_4H_TRIGGER",
            "expires_at": now + 3600,
            "evidence": {"structural_valid": True},
        }
        fixtures[str(i)] = {"opportunity": op, "updated_at": now}

    def populate(tx):
        tx.state.update(
            opportunities=fixtures,
            health={
                **{
                    name + "_at": now - 20
                    for name in (
                        "scan",
                        "context",
                        "risk",
                        "simulation",
                        "funding",
                        "telegram",
                        "outbox",
                        "shadow_observer",
                    )
                },
                "error": "telegram: TelegramError",
                "redis": "connected",
            },
            research={
                "status": "UNAVAILABLE",
                "http_status": 429,
                "error_code": "billing_not_active",
                "created_at": now,
                "reason": "API request failed",
                "asof": "2026-10-06",
            },
            context={
                "data_complete": True,
                "as_of": now,
                "expires_at": now + 600,
                "event_blackout": False,
                "risk_state": "neutral",
                "long_multiplier": 1,
                "short_multiplier": 1,
            },
            risk_circuits={
                k: {
                    "as_of": now,
                    "daily_loss_pct": 0,
                    "weekly_loss_pct": 0,
                    "drawdown_pct": 0,
                }
                for k in ("baseline_shadow", "ai_shadow")
            },
            risk_reference_equities={
                k: 10000 for k in ("baseline_shadow", "ai_shadow")
            },
        )

    await store.update(populate)
    snapshots = {}
    for view in VIEWS:
        first = await service.render_page(view)
        snapshots[view] = []
        for index in range(first["pages"]):
            result = await service.render_page(view, page=index)
            result["keyboard"] = controller._keyboard(
                view, page=index, pages=first["pages"]
            )
            snapshots[view].append(result)
    await store.close()
    data = json.dumps(snapshots, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apex Telegram · offline layout preview</title>
<style>
*{box-sizing:border-box}body{background:#0c1721;color:#dce5ef;font:14px -apple-system,BlinkMacSystemFont,Arial;margin:0;padding:28px}
main{max-width:940px;margin:auto;display:flex;gap:40px;align-items:flex-start}aside{max-width:340px}h1{font-size:26px;letter-spacing:-.6px;margin:0 0 14px}
p{line-height:1.65;color:#9cabbc}select{padding:12px;width:100%;background:#192b3c;color:white;border:1px solid #3c536c;border-radius:8px}small{display:block;color:#8c9aab;margin-top:18px;line-height:1.5}
.phone{width:390px;flex-shrink:0;background:#172431;border:1px solid #405366;border-radius:20px;overflow:hidden}
.bar{background:#203343;padding:17px;font-size:16px;font-weight:650}.bar span{color:#8eabbd;font-size:11px;display:block;font-weight:400;margin-top:4px}
.chat{height:670px;overflow:auto;padding:14px 10px 28px;background:#14212c}.bubble{background:#233545;border-radius:12px;padding:13px 12px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.4;font-size:14px;margin-bottom:6px}.buttons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px}button{background:#263c50;color:#e7f1fa;border:0;border-radius:7px;padding:11px 5px;font-size:12px;cursor:pointer}button:hover{background:#344e68}.composer{padding:17px;background:#203343;color:#8cabbc;font-size:13px}
@media(max-width:800px){body{padding:16px}main{display:block}aside{max-width:390px;margin:0 auto 20px}.phone{max-width:100%;margin:auto}.chat{height:660px}}
</style><main><aside><h1>Apex control centre</h1><p>Offline phone-width layout preview using the actual message and keyboard renderers.</p><select id="views" aria-label="Dashboard page"></select><small>Synthetic fixtures only. No exchange, Telegram or API connection. This previews layout, not native Telegram delivery. Setting changes are disabled.</small><p id="meta"></p></aside><div class="phone"><div class="bar">Apex · Elliott Wave<span>bot · synthetic preview</span></div><div class="chat"><div class="bubble" id="text"></div><div class="buttons" id="buttons"></div></div><div class="composer">Message · read-only preview</div></div></main><script>
const screens=DATA;const select=document.getElementById('views');for(const key of Object.keys(screens)){const o=document.createElement('option');o.value=key;o.textContent=key.replaceAll('_',' ');select.append(o)}
function show(view='dashboard',page=0){if(!screens[view])return;page=Math.max(0,Math.min(page,screens[view].length-1));const s=screens[view][page];select.value=view;document.getElementById('text').textContent=s.text;const buttons=document.getElementById('buttons');buttons.replaceChildren();for(const row of s.keyboard.inline_keyboard){for(const b of row){const el=document.createElement('button');el.textContent=b.text;el.onclick=()=>{const p=b.callback_data.split(':');if(p[0]==='pg')show(p[1],Number(p[2]));else if(p[0]==='v')show(p[1]);};buttons.append(el)}}document.getElementById('meta').textContent=`Page ${page+1}/${screens[view].length} · ${s.text.length} characters · ${s.keyboard.inline_keyboard.flat().length} buttons`;document.querySelector('.chat').scrollTop=0}
select.onchange=()=>show(select.value);show();</script></html>""".replace(
        "DATA", data
    )
    Path(__file__).with_name("index.html").write_text(html)
    print(
        json.dumps(
            {
                key: {"pages": len(value), "first_page_chars": len(value[0]["text"])}
                for key, value in snapshots.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(build())
