"""Capture the existing read-only dashboards; never launch experiments or open user tabs."""
import argparse, asyncio, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[2]
DEST=Path(__file__).resolve().parent/'screenshots'
CACHE=ROOT/'.cache/dashboard_docs'
CAPTURES=[]
FREEZE="""() => { for (const id of window.__documentationIntervals || []) clearInterval(id); }"""
INIT="""window.__documentationIntervals=[]; const original=setInterval; window.setInterval=(...args)=>{const id=original(...args);window.__documentationIntervals.push(id);return id;};"""

async def ready(page):
    await page.locator('#comparison-summary .metric-value').first.wait_for()
    await page.wait_for_function("document.querySelector('#fid-table tbody tr') && document.querySelector('#rbf-stages .stage-card')")
    await page.evaluate(FREEZE)

async def images(page):
    await page.evaluate("""async () => {
      const images=[...document.querySelectorAll('img')].filter(i=>i.getClientRects().length);
      for(const i of images)i.loading='eager';
      await Promise.all(images.map(i=>i.decode().catch(()=>{})));
      await document.fonts.ready;
    }""")

async def shot(page,name,selector=None,caption='',redaction=None):
    await images(page)
    target=page.locator(selector) if selector else page
    if selector:await target.scroll_into_view_if_needed()
    await target.screenshot(path=str(DEST/(name+'.png')),animations='disabled')
    await target.screenshot(path=str(CACHE/(name+'.jpg')),type='jpeg',quality=48,animations='disabled')
    data=(DEST/(name+'.png')).read_bytes()
    CAPTURES.append(dict(file=name+'.png',caption=caption,redaction=redaction,
                         sha256=hashlib.sha256(data).hexdigest(),
                         captured_utc=datetime.now(timezone.utc).isoformat()))
    print('Captured',name,flush=True)

async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sit-url',default='http://127.0.0.1:8765')
    p.add_argument('--rf-url',default='http://127.0.0.1:8766')
    args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1600,'height':1100},device_scale_factor=1,locale='ko-KR',timezone_id='UTC')
        await context.add_init_script(INIT)
        page=await context.new_page()
        failures=[]
        page.on('pageerror',lambda e:failures.append(str(e)))
        await page.goto(args.sit_url,wait_until='networkidle')
        await ready(page)
        await page.locator('[data-cfg="0"]').click()
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'01-sit-comparison',caption='SiT: CFG 0.0, 11 NFE, three samplers; completed records.')
        await page.locator('[data-nfe="6"]').click()
        await shot(page,'02-sit-previews','#previews',caption='SiT: CFG 0.0, NFE 6; three stored preview grids, latest requested RBF retraining.')
        await page.locator('#previews .preview-button').first.click()
        await page.locator('#preview-dialog[open]').wait_for()
        await shot(page,'03-preview-dialog','#preview-dialog',caption='Actual image enlargement dialog; DPM-Solver++, CFG 0.0, NFE 6.')
        await page.locator('#close-preview').click()
        await page.locator('#all-results').evaluate('(e)=>e.open=true')
        await page.locator('#all-results').scroll_into_view_if_needed()
        await page.evaluate("window.scrollTo(0,document.querySelector('#all-results').getBoundingClientRect().top+window.scrollY-24)")
        await shot(page,'04-condition-details',caption='Expanded condition details; first rows of the full 81-condition table.')
        await page.locator('[data-view="rbf"]').click()
        await page.locator('[data-rbf-cfg="0"]').click()
        await page.wait_for_function("document.querySelector('#coefficient-figure img')?.getAttribute('src').startsWith('/rbf-coefficients/0/')")
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'05-rbf-pipeline',caption='SiT CFG 0.0: 128 targets, 220 optimization repetitions over 11 NFE, 110000 RBF evaluation images.')
        await page.locator('[data-coefficient-nfe="6"]').click()
        await shot(page,'06-learned-coefficients','#coefficient-panel',caption='SiT CFG 0.0 NFE 6: log gamma and predictor/corrector CMR; unused stages omitted.')
        await page.locator('#coefficient-panel details').evaluate('(e)=>e.open=true')
        await shot(page,'07-signed-coefficients','#coefficient-panel details',caption='Signed integration coefficients and FM time increments; displayed rounded values.')
        await page.locator('[data-view="gpu"]').click()
        # Hide actual device model and memory specification in this capture context only.
        await page.evaluate("""() => {
          for(const card of document.querySelectorAll('#gpu-cards article')) {
            const model=card.querySelector('.panel-head p');if(model)model.textContent='CUDA GPU · 장치명 비공개';
            const memory=card.querySelector('.gpu-stats strong');if(memory)memory.textContent='사용량 / 장치 용량 · 비공개';
          }
          for(const dd of document.querySelectorAll('#gpu-policy dd')) {
            if(dd.previousElementSibling?.textContent==='현재 측정 기록')dd.textContent='환경별 실측 기록 · 수치 비공개';
          }
        }""")
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'08-gpu-status',caption='Actual GPU status view; device model, memory specification and measured capacity values masked for publication.',
                   redaction='DOM-only masking of device model, VRAM figures and capacity range. Application source and experiment data are unchanged.')
        await page.locator('[data-view="settings"]').click()
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'09-experiment-settings',caption='Read-only comparison, target, optimization and FID validation settings.')
        await page.goto(args.rf_url,wait_until='networkidle')
        await ready(page)
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'10-1rf-comparison',caption='1-RF CIFAR-10: unconditional, 33 completed conditions, 50000 images per condition.')
        await page.locator('[data-view="rbf"]').click()
        await page.evaluate('window.scrollTo(0,0)')
        await shot(page,'11-1rf-pipeline',caption='1-RF: 128 targets, one 128-pair fit per NFE, 550000 RBF evaluation images.')
        if failures:raise RuntimeError('Browser JavaScript errors: '+repr(failures))
        (DEST.parent/'screenshots.json').write_text(json.dumps(dict(viewport={'width':1600,'height':1100},
            source='Live dashboards backed by completed experimental records; no simulated runs',
            privacy='No browser address bar or private host address is captured; GPU mask is declared per image',
            screenshots=CAPTURES),indent=2,ensure_ascii=False)+'\n')
        await browser.close()
if __name__=='__main__':asyncio.run(main())
