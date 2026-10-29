"""Kiểm tra hợp đồng của ứng viên benchmark trong process riêng."""
import os
import subprocess
import sys
from pathlib import Path


def test_default_v2_uncertain_and_history_contract():
    code = '''
from scripts.prompt_ab import install_v2
from src.agent import perception as p, fusion as f, vlm_diagnostician as v
from src.agent.state import HistoryItem, TreatmentPlan
assert p.PROMPT_VERSION == 'perceive-v2'
before = (p.PERCEIVE_PROMPT, v.SYSTEM_PROMPT)
install_v2()
install_v2()
assert before == (p.PERCEIVE_PROMPT, v.SYSTEM_PROMPT)
assert 'uncertain' in p.VERIFY_RESPONSE_SCHEMA['properties']['answers']['items']['properties']['verdict']['enum']
assert p.parse_verify_answers({'answers':[{'hypothesis_id':'h1','verdict':'uncertain'}]}) == {}
answer = p.parse_verify_answers({'answers':[{'hypothesis_id':'h1','verdict':'defect','confidence':0.8}]})
assert answer['h1'].verdict == 'defect'
h = HistoryItem(iteration=1, plan=TreatmentPlan(iteration=1, reasoning='test', actions=[]), metrics_before={}, metrics_after={}, eval_score={}, decision='RE_PROCESS')
text = v._build_history_feedback([h])
assert 'executed' in text and 'review' in text and 'rolled_back' in text
assert v._build_history_feedback([]) == ''
'''
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
