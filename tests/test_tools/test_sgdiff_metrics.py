import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from tools import sgdiff_metrics as metrics


class TestSGDiffMetrics(unittest.TestCase):

    def test_fid_noncommuting_covariances(self):
        generated = np.array([[0., 1.], [3., -1.], [1., 4.], [-2., 0.]])
        reference = generated @ np.array([[2., -1.], [.4, 1.]]) + 3
        cov_g = np.cov(generated, rowvar=False)
        cov_r = np.cov(reference, rowvar=False)
        eigenvalues, vectors = np.linalg.eigh(cov_g)
        root_g = (vectors * np.sqrt(eigenvalues)) @ vectors.T
        trace_root = np.sqrt(np.linalg.eigvalsh(root_g @ cov_r @ root_g)).sum()
        expected = (np.square(generated.mean(0) - reference.mean(0)).sum()
                    + np.trace(cov_g) + np.trace(cov_r) - 2 * trace_root)
        self.assertAlmostEqual(metrics._frechet_distance(generated, reference),
                               expected, places=8)
        self.assertAlmostEqual(metrics._frechet_distance(generated, generated),
                               0., places=8)

    def test_fid_one_dimension_and_insufficient_samples(self):
        features = np.array([[0.], [1.], [2.]])
        self.assertAlmostEqual(metrics._frechet_distance(features, features + 3), 9.)
        with self.assertRaises(ValueError):
            metrics._frechet_distance(features[:1], features[:1])

    def test_pairing_and_metric_outputs_without_model_downloads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as directory:
            root = Path(directory)
            samples = []
            for index, key in enumerate(('validation__same', 'top__same')):
                generated = root / f'{key}.png'
                target = root / f'target{index}.png'
                Image.new('RGB', (8, 8), (0, 0, 0)).save(generated)
                Image.new('RGB', (8, 8), (255 * index,) * 3).save(target)
                samples.append(dict(id=key, sample_id='same', target=str(target),
                                    style=str(target), prompt='测试'))
            processor = Mock()
            processor.to_dict.return_value = {'do_normalize': True}
            generated_features = np.eye(2, dtype=np.float32)
            with patch.object(metrics, '_load_clip', return_value=(Mock(), processor)), \
                    patch.object(metrics, '_extract_features', side_effect=[
                        generated_features, generated_features,
                        generated_features[::-1]]):
                result = metrics.evaluate_metrics(samples, root, root / 'out',
                                                  ('clip_i', 'ssim'), device='cpu')
            self.assertAlmostEqual(result['clip_i'], 1.)
            self.assertAlmostEqual(result['clip_i_style'], 0.)
            with (root / 'out' / 'per_sample_metrics.csv').open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['id'] for row in rows], [s['id'] for s in samples])
            self.assertEqual(float(rows[0]['ssim_full']), 1.)
            self.assertLess(float(rows[1]['ssim_full']), .01)
            self.assertNotIn('fid', rows[0])
            with patch.object(metrics, '_load_fid') as load_fid:
                result = metrics.evaluate_metrics(samples[:1], root, root / 'single',
                                                  ('fid',), device='cpu')
                load_fid.assert_not_called()
            self.assertIsNone(result['fid'])
            parsed = json.loads((root / 'single' / 'metrics.json').read_text('utf-8'))
            self.assertIn('fid_reason', parsed)

    def test_preflight_propagates_weight_errors_before_training(self):
        with patch.object(metrics, '_load_fid', side_effect=RuntimeError('权重不可用')):
            with self.assertRaisesRegex(RuntimeError, '权重不可用'):
                metrics.prepare_metrics(('fid',), 'unused')


if __name__ == '__main__':
    unittest.main()
