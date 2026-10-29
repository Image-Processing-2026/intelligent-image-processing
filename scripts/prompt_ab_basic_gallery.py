"""Goal-based blind review, with controls and limits kept separate."""
import argparse
import html
import json
from pathlib import Path

e = html.escape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "blind-key.json").read_text(encoding="utf-8"))
    blocks = []
    def select(key, label, choices):
        opts = '<option value="">Chưa chấm</option>' + ''.join(f'<option>{e(x)}</option>' for x in choices)
        return f'<label>{e(label)}<select data-key="{e(key)}">{opts}</select></label>'
    for group, title in [("basic", "12 ca sửa lỗi cơ bản"), ("control", "3 ca đối chứng: bảo toàn ảnh / phong cách"), ("boundary", "2 ca giới hạn: không đòi khôi phục chi tiết đã mất")]:
        blocks.append(f'<h2>{title}</h2>')
        for case in protocol["cases"]:
            if case["group"] != group:
                continue
            key = case['id'] + '-0'
            if key not in mapping:
                blocks.append(f'<section><h3>{e(case["id"])}</h3><p>Đang xử lý — chưa đủ cặp X/Y.</p></section>')
                continue
            folder = 'blind/' + key
            figures = ''.join(f'<figure><figcaption>{label}</figcaption><a href="{folder}/{name}.png" target="_blank"><img loading="lazy" src="{folder}/{name}.png"></a></figure>' for label, name in [('Gốc','input'), ('X','X'), ('Y','Y')])
            ratings = []
            for label in ['X', 'Y']:
                ratings.append(select(key+'-'+label+'-goal', label+' — Đạt mục tiêu so với gốc', ['Không đạt', 'Đạt một phần', 'Đạt rõ', 'Chưa xác định']))
                ratings.append(select(key+'-'+label+'-harm', label+' — Tác dụng phụ mới so với gốc', ['Không thấy', 'Nhẹ', 'Nặng', 'Chưa xác định']))
            ratings.append(select(key, 'Sau khi chấm riêng X và Y, chọn kết quả dùng được hơn', ['X tốt hơn', 'Y tốt hơn', 'Ngang nhau', 'Cả hai không đạt', 'Chưa rõ']))
            blocks.append(f'''<section><h3>{e(case['id'])}</h3>
<p class="goal"><b>Yêu cầu đưa cho cả hai phiên bản:</b> {e(case['intent'])}</p>
<p><a href="{e(case['source_page'], quote=True)}">{e(case['author'])} · {e(case['license'])}</a></p>
<div class="images">{figures}</div>
<details><summary>Xem crop 1:1 cùng tọa độ của gốc / X / Y</summary><p>Mở từng crop để xem pixel gốc, không nội suy.</p><div class="images"><a target="_blank" href="{folder}/Original-crop.png"><img src="{folder}/Original-crop.png"></a><a target="_blank" href="{folder}/X-crop.png"><img src="{folder}/X-crop.png"></a><a target="_blank" href="{folder}/Y-crop.png"><img src="{folder}/Y-crop.png"></a></div></details>
<div class="ratings">{''.join(ratings)}</div><label>Ghi chú quan sát<input data-key="{key}-notes" placeholder="Ví dụ: mặt sáng rõ hơn nhưng nền bị cháy, chữ bị bệt..."></label>
<details><summary>Tham khảo thuật toán cố định — xem sau khi chấm X/Y</summary><p>{e(case['baseline_recipe'])}. Đây không phải ảnh chuẩn hay đáp án; có thể còn lỗi.</p><a target="_blank" href="references/{case['id']}/baseline.png"><img src="references/{case['id']}/baseline.png"></a></details></section>''')
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A/B — sửa lỗi cơ bản</title>
<style>body{font:16px system-ui;margin:24px;background:#edf0f3;color:#17212d}section{background:white;padding:22px;margin:20px 0;border-radius:12px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%;max-height:650px;object-fit:contain}figure img{width:100%;background:#eee}figcaption{font-weight:bold}.goal{padding:14px;background:#e9f3ff}.ratings{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:20px 0}label{display:block}select,input{display:block;padding:10px;margin-top:6px;max-width:100%;font:inherit}input{width:95%}details{margin-top:18px}summary{cursor:pointer}button{padding:12px;font:inherit;cursor:pointer}header{position:sticky;top:0;background:#edf0f3;padding:10px;z-index:1}@media(max-width:700px){body{margin:10px}.ratings{grid-template-columns:1fr}}</style>
<h1>Ảnh có lỗi rõ: phiên bản nào sửa tốt hơn?</h1>
<p>17 ảnh thật · 12 ca cơ bản / 3 đối chứng / 2 giới hạn · một lượt mỗi phiên bản. X/Y đổi vị trí theo từng ảnh. Không xem khóa giải mã trước khi chấm.</p>
<p><b>Cách chấm:</b> đọc yêu cầu → đối chiếu từng kết quả với ảnh gốc → chấm mức đạt mục tiêu và tác dụng phụ → mới chọn X/Y. Ở ca đối chứng, giữ nguyên đúng phong cách có thể là “Đạt rõ”. Không cần ưu tiên ảnh sửa mạnh hơn. Với nhiễu, mở ảnh/crop ở kích thước 100% để soi chi tiết.</p>
<p>Bộ này vẫn là thử nghiệm phát triển; không có ảnh chuẩn sạch. Ca sáng/màu có mức độ khác nhau, không coi tên nhóm là bằng chứng chất lượng.</p>
<header><button onclick="exportRatings()">Xuất đánh giá JSON</button> <span id="progress"></span></header>''' + '\n'.join(blocks)
    page += '<script>const runId=' + json.dumps(root.name) + ';const ns="prompt-ab:"+runId+":";'
    page += '''const fields=[...document.querySelectorAll('[data-key]')];
function progress(){document.getElementById('progress').textContent=fields.filter(f=>f.tagName==='SELECT'&&f.value).length+' / '+fields.filter(f=>f.tagName==='SELECT').length+' mục đã chấm'}
for(const f of fields){f.value=localStorage.getItem(ns+f.dataset.key)||'';f.onchange=()=>{localStorage.setItem(ns+f.dataset.key,f.value);progress()}}progress();
function exportRatings(){const data={schema_version:2,run_id:runId,ratings:Object.fromEntries(fields.map(f=>[f.dataset.key,f.value]))};const a=document.createElement('a');const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.href=u;a.download='ratings-'+runId+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
</script></html>'''
    (root / 'gallery.html').write_text(page, encoding='utf-8')
    print(f'Gallery: {len(mapping)}/17 paired cases')


if __name__ == '__main__':
    main()
