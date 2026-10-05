# Lumexa CalculiX solver on an Android tablet (Termux + Debian)

Free, no card. Needs: Termux (F-Droid build), ~2 GB free storage, the tablet awake and online while jobs run.

## 1. Install Debian inside Termux (once)
```
pkg update && pkg upgrade -y
pkg install proot-distro -y
termux-setup-storage
proot-distro install debian
```

## 2. Put the folder where Debian can see it
Copy the `analysis_service` folder into your phone's Download folder, then:
```
proot-distro login debian --bind ~/storage/downloads:/mnt/dl
```

## 3. Inside Debian (once)
```
apt update
apt install -y python3 python3-venv python3-pip calculix-ccx gmsh \
  libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libfontconfig1 libgomp1
cp -r /mnt/dl/analysis_service ~/analysis_service && cd ~/analysis_service
python3 -m venv ~/venv && . ~/venv/bin/activate
pip install -r requirements.txt
```
If `pip install gmsh` fails on this CPU, use the Debian package instead:
`apt install -y python3-gmsh` and create the venv with `--system-site-packages`.

## 4. Run the solver (every time)
```
proot-distro login debian --bind ~/storage/downloads:/mnt/dl
. ~/venv/bin/activate && cd ~/analysis_service
export CCX_API_KEY=choose-a-long-random-secret
export CCX_THREADS=4 MAX_NODES=30000
uvicorn app:app --host 0.0.0.0 --port 8000
```
Keep this Termux session open. In another Termux session run `termux-wake-lock`.

## 5. Check it (a third session, outside Debian is fine)
```
curl -s localhost:8000/health
curl -s -H "X-API-Key: choose-a-long-random-secret" localhost:8000/selftest
```
/selftest solves a cantilever and compares with beam theory: deflection ratio ~0.95-1.03,
stress ratio ~1.0-1.3, bc_check.ok true and equilibrium error under 3 %.

## 6. Make it reachable from Render (a tunnel)
```
pkg install cloudflared -y
cloudflared tunnel --url http://localhost:8000
```
It prints an https://....trycloudflare.com address (it changes each time you restart it).

## 7. Tell the main backend (Render environment)
- CCX_SERVICE_URL = the tunnel address
- CCX_API_KEY     = the same secret as above
- FEA_SOLVER      = calculix

Then check from outside: `https://v8-wa56.onrender.com/ccx-health` and `/ccx-selftest`.
