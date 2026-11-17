import os
import urllib.request
import urllib.parse
import ssl
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def download_year(year, out_dir="Dataset"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"RF25_ind{year}_rfp25.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 20 * 1024 * 1024:
        print(f"[{year}] Already exists ({os.path.getsize(out_path)/(1024*1024):.1f} MB). Skipping.")
        return True
    
    url = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
    data = urllib.parse.urlencode({"RF25": str(year)}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
    
    for attempt in range(1, 4):
        try:
            print(f"[{year}] Downloading... (attempt {attempt})")
            t0 = time.time()
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                content = resp.read()
            if len(content) < 15 * 1024 * 1024:
                raise ValueError(f"File size too small: {len(content)} bytes")
            with open(out_path, "wb") as f:
                f.write(content)
            dt = time.time() - t0
            print(f"[{year}] Saved {len(content)/(1024*1024):.1f} MB in {dt:.1f}s.")
            return True
        except Exception as e:
            print(f"[{year}] Attempt {attempt} failed: {e}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    years = list(range(2010, 2025))  # 15 years: 2010 to 2024
    print(f"Starting download for {len(years)} years: {years[0]} - {years[-1]}")
    success = []
    for yr in years:
        ok = download_year(yr)
        success.append((yr, ok))
    print("\n--- Summary ---")
    for yr, ok in success:
        print(f"Year {yr}: {'OK' if ok else 'FAILED'}")
