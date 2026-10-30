#!/data/data/com.termux/files/usr/bin/bash
# Lumexa start-up: updates the solver file from GitHub, starts CalculiX (8000), the backend (8001) and a tunnel,
# then tells Supabase the tunnel address. Secrets (KEY, REG, SB, ANON) live in ~/.lumexa.env, not in this file.
[ -f ~/.lumexa.env ] && . ~/.lumexa.env
if [ -z "$KEY" ] || [ -z "$REG" ] || [ -z "$SB" ] || [ -z "$ANON" ]; then
  echo "KEY / REG / SB / ANON missing in ~/.lumexa.env - cannot start."; exit 1
fi
GH="https://raw.githubusercontent.com/xorlumexa-creator/V8.6/main"
DEB="${PREFIX:-/data/data/com.termux/files/usr}/var/lib/proot-distro/installed-rootfs/debian"
if [ ! -d "$DEB/root" ]; then echo "Debian folder not found at $DEB"; exit 1; fi
termux-wake-lock
pkill -f proot 2>/dev/null; pkill cloudflared 2>/dev/null; sleep 3

# solver launcher, runs inside Debian: pull app.py from GitHub (kept only if valid), then start the solver
cat > "$DEB/root/solver_start.sh" <<'EOS'
#!/bin/bash
CK="$1"; GH="$2"
. ~/venv-ccx/bin/activate
cd ~/backend/analysis_service || { echo "solver folder missing"; exit 1; }
cp -f app.py app.py.bak 2>/dev/null
if curl -fsSL "$GH/analysis_service/app.py" -o app.new && python3 -m py_compile app.new && grep -q v2-adaptive-mesh app.new; then
  mv -f app.new app.py; echo "solver file: updated from GitHub"
else
  rm -f app.new; echo "solver file: GitHub copy missing or invalid - keeping the current one"
fi
export CCX_API_KEY="$CK" CCX_THREADS=4 MAX_NODES=30000 FEM_STRAIGHT_MIDSIDE=1
exec uvicorn app:app --host 0.0.0.0 --port 8000
EOS
nohup proot-distro login debian -- bash /root/solver_start.sh "$KEY" "$GH" > ~/ccx.log 2>&1 &
sleep 10
nohup proot-distro login debian -- bash -c ". ~/venv-backend/bin/activate && cd ~/backend && export FEA_SOLVER=calculix CCX_SERVICE_URL=http://127.0.0.1:8000 CCX_API_KEY=$KEY MALLOC_ARENA_MAX=2 && exec uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1" > ~/backend.log 2>&1 &
c1=000; c2=000
for i in $(seq 1 60); do
  c1=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)
  c2=$(curl -s -o /dev/null -w '%{http_code}' localhost:8001/)
  [ "$c1" = 200 ] && [ "$c2" = 200 ] && break
  sleep 3
done
echo "CalculiX 8000: $c1 | backend 8001: $c2"
grep -a 'solver file' ~/ccx.log
echo "solver version: $(curl -s localhost:8000/health | grep -o '"version":"[^"]*"')"
if [ "$c2" != 200 ]; then echo "Backend did not start. Send me: tail -20 ~/backend.log"; exit 1; fi
nohup cloudflared tunnel --url http://localhost:8001 > ~/tunnel8001.log 2>&1 &
URL=""
for i in $(seq 1 20); do
  URL=$(grep -a -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tunnel8001.log | tail -1)
  [ -n "$URL" ] && break
  sleep 2
done
echo "TUNNEL: $URL"
curl -s -X POST "$SB/rest/v1/rpc/lumexa_register_backend" -H "apikey: $ANON" -H "Authorization: Bearer $ANON" -H "Content-Type: application/json" -d "{\"p_url\":\"$URL\",\"p_secret\":\"$REG\"}"
echo
