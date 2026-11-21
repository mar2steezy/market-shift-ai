"""
alerts.py - free phone alerts through ntfy.sh (no account, no pip install).

Setup (once):
  1. Install the free "ntfy" app on your phone and subscribe to your private topic name.
  2. In the bot's folder, create a file ntfy_topic.txt containing ONLY that topic name:
         echo your-topic-name>ntfy_topic.txt
  3. Test it:   python alerts.py

If ntfy_topic.txt does not exist, alerts are silently skipped, so the bots still work.
Treat the topic name like a password: anyone who knows it can read your alerts.
Alerts only ever contain the symbol and what the bot did. Never balances or keys.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _topic():
    path = os.path.join(HERE, "ntfy_topic.txt")
    if os.path.exists(path):
        t = open(path, encoding="utf-8").read().strip()
        if t:
            return t
    return os.environ.get("NTFY_TOPIC", "").strip()


def notify(title, message, priority="default"):
    """priority: min, low, default, high, urgent. Never raises; the bot must never crash over an alert."""
    topic = _topic()
    if not topic:
        return False
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), method="POST")
        req.add_header("Title", title.encode("ascii", "ignore").decode())
        req.add_header("Priority", priority)
        urllib.request.urlopen(req, timeout=8).read()
        return True
    except Exception as ex:
        print(f"(phone alert not sent: {ex})")
        return False


if __name__ == "__main__":
    if not _topic():
        sys.exit("No topic found. Create ntfy_topic.txt first (see the notes at the top of this file).")
    ok = notify("Trading bot test", "If you can read this on your phone, alerts are working.", "default")
    print("Sent. Check your phone." if ok else "Failed to send. Check the topic name and your internet connection.")
