"""不加载模型权重，验证 BF 测试清单与 Mymodel 的样本对应关系。"""

import importlib.util
import json
import random
import shutil
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'sgdiff_test_entry', PROJECT_ROOT / 'tools' / 'test_sgdiff.py')
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class TestCollectSamples(unittest.TestCase):

    def setUp(self):
        # 普通 mkdir 避免 Python 3.13 临时目录在 Windows 沙箱中的 ACL 限制。
        self.root = PROJECT_ROOT / ('_sgdiff_test_' + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def make_sample(self, split, stem, category=None):
        root = self.root / split
        if category:
            root /= category
        for folder, suffix, content in (
                ('gt', '.jpg', b'image'),
                ('texture', '.png', b'texture'),
                ('text', '.txt', b'  a red\n dress  ')):
            path = root / folder / (stem + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return root

    def test_full_test_keeps_category_order_and_excludes_bags(self):
        # 同名 stem 可出现在不同类别；各类文件扩展名也不必一致。
        for category in ('bag', 'dress', 'pants', 'outwear', 'top'):
            for stem in ('z_last', 'a_first'):
                self.make_sample('test', stem, category)

        samples = ENTRY.collect_samples(self.root, 'test', max_samples=0)

        self.assertEqual(
            [sample['id'] for sample in samples],
            [category + '__' + stem
             for category in ('top', 'outwear', 'pants', 'dress')
             for stem in ('a_first', 'z_last')])
        self.assertEqual(
            [sample['sample_id'] for sample in samples],
            ['%06d' % index for index in range(8)])
        self.assertTrue(all(sample['prompt'] == 'a red dress' for sample in samples))

    def test_fixed_subset_matches_mymodel_index_shuffle(self):
        stems = ['dress_%02d' % index for index in range(12)]
        for stem in reversed(stems):
            self.make_sample('validation', stem)

        # Mymodel 打乱排序后数据的下标，再对选定子集重新编号。
        indices = list(range(len(stems)))
        random.Random(42).shuffle(indices)
        expected = ['validation__' + stems[index] for index in indices[:5]]
        first = ENTRY.collect_samples(self.root, max_samples=5, split_seed=42)
        random.seed(999)
        second = ENTRY.collect_samples(self.root, max_samples=5, split_seed=42)

        self.assertEqual([sample['id'] for sample in first], expected)
        self.assertEqual(first, second)
        self.assertEqual(
            [sample['sample_id'] for sample in first],
            ['000000', '000001', '000002', '000003', '000004'])

    def test_mymodel_manifest_keeps_order_and_generation_seed_ids(self):
        for stem in ('alpha', 'beta', 'gamma'):
            self.make_sample('validation', stem)
        source = [dict(sample_id=sample_id, prompt='prompt for ' + stem,
                       target='gt/' + stem + '.jpg',
                       texture='texture/' + stem + '.png', category='validation')
                  for stem, sample_id in [('gamma', '000105'), ('alpha', '000008'),
                                          ('beta', '000042')]]
        split_file = self.root / 'mymodel_split.json'
        split_file.write_text(json.dumps(source), encoding='utf-8-sig')

        samples = ENTRY.collect_samples(
            self.root, max_samples=2, split_seed=987, split_file=split_file)

        self.assertEqual(
            [sample['id'] for sample in samples],
            ['validation__gamma', 'validation__alpha'])
        # 即使只取部分清单，也保留原 sample_id，确保后续 seed + sample_id 不漂移。
        self.assertEqual([sample['sample_id'] for sample in samples], ['000105', '000008'])
        self.assertEqual([42 + int(sample['sample_id']) for sample in samples], [147, 50])
        self.assertEqual(samples[0]['prompt'], 'prompt for gamma')
        self.assertEqual(
            Path(samples[0]['target']), self.root / 'validation' / 'gt' / 'gamma.jpg')

    def test_missing_pair_is_rejected_before_generation(self):
        root = self.make_sample('validation', 'incomplete')
        (root / 'texture' / 'incomplete.png').unlink()

        with self.assertRaisesRegex(ValueError, '完整配对'):
            ENTRY.collect_samples(self.root, max_samples=0)


if __name__ == '__main__':
    unittest.main()
