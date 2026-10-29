"""Decode schema-2 human ratings and audit against actual output pixels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('ratings', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    original = json.loads(args.ratings.read_text(encoding='utf-8-sig'))
    assert original['schema_version'] == 2 and original['run_id'] == root.name
    ratings = original['ratings']
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    mapping = json.loads((root/'blind-key.json').read_text(encoding='utf-8'))
    rows = []
    allowed_goal = {'Không đạt', 'Đạt một phần', 'Đạt rõ', 'Chưa xác định'}
    allowed_harm = {'Không thấy', 'Nhẹ', 'Nặng', 'Chưa xác định'}
    for case in protocol['cases']:
        key = case['id']+'-0'
        row = {'id':case['id'], 'group':case['group'], 'mapping':mapping[key],
               'preference_raw':ratings[key], 'notes':ratings[key+'-notes']}
        arrays = []
        for label, ver in mapping[key].items():
            goal, harm = ratings[key+'-'+label+'-goal'], ratings[key+'-'+label+'-harm']
            assert goal in allowed_goal and harm in allowed_harm
            row[ver] = {'goal':goal, 'harm':harm}
            arrays.append(np.asarray(Image.open(root/case['id']/'0'/ver/'output.png').convert('RGB')))
        row['identical_pixels'] = bool(np.array_equal(*arrays))
        pref = row['preference_raw']
        row['winner_raw'] = mapping[key][pref[0]] if pref in ['X tốt hơn','Y tốt hơn'] else None
        row['winner_audited'] = None if row['identical_pixels'] else row['winner_raw']
        row['audit_flags'] = []
        if row['identical_pixels'] and row['winner_raw']:
            row['audit_flags'].append('Directional preference on identical pixels: excluded from wins')
        if row['identical_pixels'] and row['v1'] != row['v2']:
            row['audit_flags'].append('Different per-output ratings on identical pixels: unresolved, do not infer goal advantage')
        rows.append(row)
    shutil.copyfile(args.ratings, root/'human-ratings-original.json')
    payload = {'source_sha256':hashlib.sha256(args.ratings.read_bytes()).hexdigest(), 'rows':rows}
    (root/'human-ratings-decoded.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Chấm mù của người dùng — V1/V2 trên 17 ảnh', '',
             'Đủ 85/85 trường chấm; 17 ảnh độc lập, một lượt/phiên bản, một người chấm. Giữ nguyên dữ liệu gốc và giải mã riêng.', '',
             '## Kết quả có thể dùng', '',
             'Trong 7 ca cơ bản có đầu ra khác nhau: V2 được chọn 4, V1 được chọn 1, cả hai không đạt 1, chưa rõ 1.',
             'V2 được chấm Đạt rõ ở mèo tối, băng, ảnh bạc màu, mặt người cầm micro. V1 không đạt trên cả bốn ca này.',
             'V2 thua ở hoa tối: Không đạt / tác dụng phụ Nặng; V1 Đạt một phần / tác dụng phụ Nhẹ.',
             'Ca bút nhiễu: cả hai Không đạt. Ca chậu cây nhiễu: chưa xác định mục tiêu và chưa rõ ưu tiên.',
             'Ba đối chứng: cả hai Đạt rõ, không thấy tác dụng phụ, đầu ra giữ nguyên. Hai ca giới hạn chưa xác định đạt mục tiêu.', '',
             '## Kiểm tra tính nhất quán', '',
             'Ảnh đền và đồ chơi xanh: X/Y giống hệt từng pixel và giống ảnh gốc. Phiếu Y tốt hơn ở hai ca này không cung cấp bằng chứng thắng/thua. Riêng ảnh đền còn có điểm mục tiêu khác nhau trên cùng pixel; đánh dấu chưa nhất quán, không tự sửa phiếu.',
             'Tất cả 7 phiếu ưu tiên có hướng đều chọn Y. Vị trí có thể ảnh hưởng đánh giá; đây là khả năng cần kiểm tra, chưa đủ để kết luận có thiên lệch. Nếu chấm lại nên đảo vị trí cùng đầu ra và ẩn lựa chọn cũ.', '',
             '| Nhóm | V1 được chọn (thô) | V2 được chọn (thô) | V1 sau kiểm tra pixel | V2 sau kiểm tra pixel |',
             '|---|---:|---:|---:|---:|']
    for group in ['basic','control','boundary']:
        rs=[r for r in rows if r['group']==group]
        lines.append('| '+group+' | '+' | '.join(str(sum(r[field]==v for r in rs)) for field in ['winner_raw','winner_audited'] for v in ['v1','v2'])+' |')
    lines += ['', '## Chi tiết từng ảnh', '',
              '| Ảnh | Nhóm | V1 mục tiêu / tác dụng phụ | V2 mục tiêu / tác dụng phụ | Ưu tiên giải mã | Kiểm tra |',
              '|---|---|---|---|---|---|']
    for row in rows:
        pref=row['winner_raw'] or row['preference_raw']
        audit='Không tính thắng/thua: cùng pixel' if row['audit_flags'] else ('Cùng pixel' if row['identical_pixels'] else 'Khác pixel')
        cells=[row['id'],row['group']]+[row[v]['goal']+' / '+row[v]['harm'] for v in ['v1','v2']]+[pref,audit]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Diễn giải và giới hạn', '',
              'V2 có tín hiệu cải thiện ở một số ca thiếu sáng/tương phản, đồng thời có một ca suy giảm rõ khi chỉnh vùng hoa. Chưa có cơ sở thay mặc định sản phẩm bằng toàn bộ V2.',
              'Không đổi số lần được chọn thành tỷ lệ thành công chung: mẫu nhỏ, có ca chưa xác định, chưa lặp theo ảnh, chưa thử Clarify/Refine qua hội thoại và chưa ablation từng thay đổi prompt.',
              'Ở ca hoa, trace V2 cho thấy ba vòng gamma vùng hoa 1.6, gamma 1.2, rồi CLAHE 1.2; reviewer đều REVISE vì black_lift. Cần kiểm tra cơ chế chỉnh trên ảnh đã sửa và chọn/khôi phục kết quả trước khi coi việc giảm tham số là đã khắc phục lỗi.',
              'Intent ở thí nghiệm này được truyền bằng notes hiện hữu. Nhánh không có actionable defect có thể dừng trước khi Plan đọc notes, nên giữ nguyên ảnh chưa chứng minh model đã đọc rồi từ chối yêu cầu người dùng.']
    (root/'human-ratings-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'cases':len(rows),'basic_different_pixels':sum(r['group']=='basic' and not r['identical_pixels'] for r in rows),'audited_wins':dict(Counter(r['winner_audited'] for r in rows if r['winner_audited'])),'flagged':[r['id'] for r in rows if r['audit_flags']]},ensure_ascii=False))


if __name__=='__main__':
    main()
