"""检查扩展指标输出与缺失值处理，不下载指标权重。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools import evaluate_bf_extended as entry


class TestExtendedMetrics(unittest.TestCase):
    def test_summary_does_not_turn_missing_colors_into_zero(self):
        result = entry.summarize([{'prompt_color_delta_e': None},
                                  {'prompt_color_delta_e': float('nan')},
                                  {'prompt_color_delta_e': 12.0}])
        self.assertEqual(result['prompt_color_delta_e_valid'], 1)
        self.assertEqual(result['prompt_color_delta_e_mean'], 12.0)
        self.assertIsNone(result['struct_iou_mean'])

    def test_complete_output_with_mock_metric_models(self):
        names = ('eval', 'eval.metrics', 'eval.distribution_diagnostics',
                 'eval.eval_utils', 'color_conflict_utils', 'garment_mask_utils')
        modules = {name: ModuleType(name) for name in names}
        modules['eval'].metrics = modules['eval.metrics']
        metrics = modules['eval.metrics']
        metrics.evaluate_full = lambda *a, **k: {key: 0.5 for key in entry.PAIR_KEYS}
        metrics.compute_clip_i_values = lambda *a, **k: [0.8]
        metrics._clip_model_cache = {}
        modules['eval.distribution_diagnostics'].compute_distribution_metrics = (
            lambda *a, **k: dict(legacy_fid=10., kid_mean=0.01, kid_std=0.002))
        utils = modules['eval.eval_utils']
        utils.prepare_evaluation_masks = lambda *a, **k: {'garment': np.ones((8, 8), dtype=bool)}
        utils.json_safe = lambda value: value
        color = modules['color_conflict_utils']
        color.extract_text_color = lambda prompt: (None, None)
        color.dominant_rgb_from_pil = lambda *a: (20, 20, 20)
        color.delta_e_rgb = lambda *a: 0.
        modules['garment_mask_utils'].mask_backend_info = lambda: {'mask_backend': 'test'}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as tmp:
            root = Path(tmp)
            for name in ('eval/metrics.py', 'eval/eval_utils.py', 'eval/distribution_diagnostics.py',
                         'garment_mask_utils.py', 'color_conflict_utils.py'):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('', encoding='utf-8')
            for folder in ('sketch', 'gt', 'generated', 'real', 'texture'):
                (root / folder).mkdir()
                Image.new('RGB', (8, 8), 'white').save(root / folder / 'a.png')
            (root / 'manifest.json').write_text(json.dumps({'samples': [dict(
                id='a', sample_id='000000', prompt='a shirt', target=str(root / 'gt/a.png'))]}))
            with patch.dict('sys.modules', modules), patch.object(entry.sys, 'path', list(entry.sys.path)), \
                    patch.object(entry.sys, 'argv', ['eval', '--output-dir', str(root),
                                                     '--mymodel-root', str(root), '--device', 'cpu']):
                entry.main()
            result = json.loads((root / 'metrics_extended.json').read_text(encoding='utf-8'))
            self.assertEqual(result['fid'], 10.)
            self.assertEqual(result['kid'], 0.01)
            self.assertEqual(result['clip_i_real_mean'], 0.8)
            self.assertEqual(result['prompt_color_delta_e_valid'], 0)
            self.assertTrue((root / 'per_sample_extended.csv').is_file())


if __name__ == '__main__':
    unittest.main()
