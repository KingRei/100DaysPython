import io, json, os, re, sys
KIT = os.path.dirname(os.path.abspath(__file__))
REPO = '/sessions/compassionate-dazzling-euler/mnt/100DaysPython'

def build(day_js_path, out_path):
    day = io.open(day_js_path, encoding='utf-8').read()
    meta = {}
    for k in ['DAY','TITLE_ZH','TITLE_EN','SUB_ZH','SUB_EN','FOLDER','MEDIUM']:
        m = re.search(r'^//\s*' + k + r':\s*(.*)$', day, re.M)
        assert m, k + ' missing in ' + day_js_path
        meta[k] = m.group(1).strip()
    head = io.open(os.path.join(KIT, 'shell_head.html'), encoding='utf-8').read()
    for k, v in meta.items():
        head = head.replace('{{' + k + '}}', v)
    common = io.open(os.path.join(KIT, 'common.js'), encoding='utf-8').read()
    tail = io.open(os.path.join(KIT, 'shell_tail.html'), encoding='utf-8').read()
    html = head + '<script>\n' + day + '\n</script>\n<script>\n' + common + '\n</script>\n' + tail
    io.open(out_path, 'w', encoding='utf-8', newline='\n').write(html)
    # the js-only bundle for headless testing
    js = day + '\n' + common
    io.open(out_path + '.js', 'w', encoding='utf-8', newline='\n').write(js)
    return meta, len(html)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        meta, n = build(p, os.path.join(KIT, 'out', os.path.basename(p).replace('.js', '.html')))
        print('built day', meta['DAY'], n, 'bytes')
