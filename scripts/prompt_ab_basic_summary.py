"""Summarize measured operations separately from uncollected human ratings."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    root = p.parse_args().run
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'comparison.json').read_text(encoding='utf-8'))
    cases = {c['id']: c for c in protocol['cases']}
    lines = ['# Kết quả vận hành — bộ ảnh cơ bản', '',
             f"Hoàn tất {len(rows)}/34 lượt trên 17 ảnh; một lượt mỗi phiên bản. Chưa có điểm chất lượng từ người dùng.",
             'Số ảnh thay đổi không phải số ảnh sửa tốt. Calls là số lời gọi qua wrapper các phase; thời gian chịu ảnh hưởng hai cặp chạy đồng thời.', '',
             '| Nhóm | Ver | Lượt | Ảnh đổi pixel | Lỗi API | Fallback plan | Calls | Giây trung vị |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ['basic','control','boundary']:
        for version in ['v1','v2']:
            rs = [r for r in rows if r['version']==version and cases[r['image']]['group']==group]
            if rs:
                lines.append(f"| {group} | {version} | {len(rs)} | {sum(r['changed'] for r in rs)} | {sum(r['api_errors'] for r in rs)} | {sum(r['plan_fallbacks'] for r in rs)} | {sum(r['calls'] for r in rs)} | {statistics.median(r['seconds'] for r in rs):.1f} |")
    lines += ['', '| Ảnh | Nhóm | V1 đổi pixel | V2 đổi pixel | X/Y cùng pixel |', '|---|---|---|---|---|']
    for id, case in cases.items():
        rs = {r['version']:r for r in rows if r['image']==id}
        if len(rs)==2:
            lines.append(f"| {id} | {case['group']} | {rs['v1']['changed']} | {rs['v2']['changed']} | {rs['v1']['output_sha256']==rs['v2']['output_sha256']} |")
    lines += ['', '## Nhánh dừng trước Plan', '',
              'Các lượt dưới đây không gọi LLM Plan. Notes không được đưa vào Perceive/Verify trong mã đang thử. Đây là giới hạn của phép thử pipeline với ghi chú, không chứng minh model đã đọc rồi từ chối yêu cầu.', '',
              '| Ảnh | Ver | Preserve cuối | Kết cục |', '|---|---|---|---|']
    for row in rows:
        if row['phase_calls']['plan']:
            continue
        state=json.loads((root/row['image']/str(row['repeat'])/row['version']/'state.json').read_text(encoding='utf-8'))
        preserve=', '.join(p['aspect']+'@'+p['region'] for p in (state.get('diagnosis') or {}).get('preserve',[]))
        lines.append(f"| {row['image']} | {row['version']} | {preserve} | {row['outcome']} |")
    lines += ['', 'Xem `comparison.md` và trace để phân biệt skip, rollback, lỗi API và giữ nguyên. Không mở blind-key.json trước khi chấm.',
              'Chưa kết luận V2 tốt hơn. Cần chấm đạt mục tiêu/tác dụng phụ theo từng ảnh; ảnh giống hệt chỉ tính hòa kỹ thuật, không suy ra cả hai đạt.',
              'Ca giới hạn còn cần đọc thông báo/trace để chấm nhận biết giới hạn; gallery ảnh chỉ đánh giá phần thị giác.']
    (root/'technical-summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Summarized {len(rows)} runs')


if __name__=='__main__':
    main()
