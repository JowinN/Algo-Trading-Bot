#!/bin/bash

BOT_SVC="mudrex-bot"
DASH_SVC="mudrex-dash"

case "$1" in

  start)
    sudo systemctl start "$BOT_SVC" "$DASH_SVC"
    echo "🚀 Bot started"
    echo "📊 Dashboard started  →  http://localhost:5000"
    ;;

  stop)
    sudo systemctl stop "$BOT_SVC" "$DASH_SVC"
    echo "🛑 Bot stopped"
    echo "🛑 Dashboard stopped"
    ;;

  restart)
    sudo systemctl restart "$BOT_SVC" "$DASH_SVC"
    echo "🔄 Bot restarted"
    echo "🔄 Dashboard restarted  →  http://localhost:5000"
    ;;

  status)
    echo "── BOT ─────────────────────────────────────"
    sudo systemctl status "$BOT_SVC" --no-pager -l
    echo ""
    echo "── DASHBOARD ───────────────────────────────"
    sudo systemctl status "$DASH_SVC" --no-pager -l
    ;;

  logs)
    journalctl -u "$BOT_SVC" -f
    ;;

  dashlogs)
    journalctl -u "$DASH_SVC" -f
    ;;

  enable)
    sudo systemctl enable "$BOT_SVC" "$DASH_SVC"
    echo "✅ Auto-start on boot enabled"
    ;;

  disable)
    sudo systemctl disable "$BOT_SVC" "$DASH_SVC"
    echo "⛔ Auto-start on boot disabled"
    ;;

  *)
    echo "Usage: ./run.sh {start|stop|restart|status|logs|dashlogs|enable|disable}"
    exit 1
    ;;

esac
