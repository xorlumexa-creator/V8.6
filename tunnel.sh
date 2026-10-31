#!/data/data/com.termux/files/usr/bin/bash
# Restarts ONLY the tunnel and registers the new address with Supabase. It never touches the solver or the backend,
# so a job that is running keeps running. (start_all.sh restarts everything and loses running jobs - do not use it mid-job.)
[ -f ~/.lumexa.env ] && . ~/.lumexa.env
if [ -z "$SB" ] || [ -z "$ANON" ] || [ -z "$REG" ]; then echo "SB / ANON / REG missing in ~/.lumexa.env"; exit 1; fi
c2=$(curl -s -o /dev/null -w '%{http_code}' localhost:8001/)
if [ "$c2" != 200 ]; then echo "Backend is NOT running on 8001 (got $c2). Run start_all.sh instead."; exit 1; fi
echo "backend is up - restarting only the tunnel"
pkill cloudflared 2>/dev/null; sleep 2
nohup cloudflared tunnel --url http://localhost:8001 > ~/tunnel8001.log 2>&1 &
URL=""
for i in $(seq 1 20); do
  URL=$(grep -a -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tunnel8001.log | tail -1)
  [ -n "$URL" ] && break
  sleep 2
done
echo "TUNNEL: $URL"
[ -z "$URL" ] && { echo "no tunnel address yet - run it again"; exit 1; }
curl -s -X POST "$SB/rest/v1/rpc/lumexa_register_backend" -H "apikey: $ANON" -H "Authorization: Bearer $ANON" -H "Content-Type: application/json" -d "{\"p_url\":\"$URL\",\"p_secret\":\"$REG\"}"
echo
