"""Critical invariants: architecture, splits, evaluation and adaptive schedules."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_utils import make_mlp, adaptive_dropout
from utils.experiment_utils import evaluate, loaders, parser, split_indices, run


class ExperimentsTest(unittest.TestCase):
    def test_invalid_experiment_settings(self):
        for setting in (['--epochs','0'], ['--batch-size','1'], ['--train-limit','-1'], ['--test-limit','-1']):
            args=parser('test').parse_args(['--smoke',*setting])
            with self.assertRaises(ValueError):
                run(args,'mnist','invalid',lambda:make_mlp(784,[8]),'invalid')

    def test_train_validation_disjoint(self):
        for size in (50000, 60000):
            train, val = split_indices(size, 42, 6000)
            self.assertFalse(set(train) & set(val))
            self.assertEqual((len(train), len(val)), (6000, 1200))
            self.assertEqual((train, val), split_indices(size, 42, 6000))

    def test_depth_and_gradients(self):
        for depth in (1, 2, 3, 5, 7):
            model = make_mlp(16, [8] * (depth - 1), .3, depth > 1)
            self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.modules()), depth)
            out = model(torch.randn(4, 1, 4, 4))
            self.assertEqual(tuple(out.shape), (4, 10))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_adaptive_dropout(self):
        model = make_mlp(16, [8, 8, 8], .1)
        adaptive_dropout(model, 4, 5)
        values = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]
        self.assertEqual(values[-1], .5)
        self.assertTrue(values[0] < values[1] < values[2])

    def test_evaluation_is_weighted_and_has_no_gradients(self):
        model = nn.Linear(2, 2)
        x = torch.randn(5, 2); y = torch.tensor([0, 1, 0, 1, 0])
        metrics, predictions, targets = evaluate(model, DataLoader(TensorDataset(x,y), batch_size=3), 'cpu')
        self.assertAlmostEqual(metrics['loss'], nn.functional.cross_entropy(model(x), y).item(), places=6)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(targets, y.tolist())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reproducible_loaders(self):
        args = parser('test').parse_args(['--smoke'])
        a, _ = loaders(args, 'mnist'); b, _ = loaders(args, 'mnist')
        self.assertTrue(torch.equal(next(iter(a[0]))[0], next(iter(b[0]))[0]))
        self.assertNotEqual(a[0].dataset.tensors[0].data_ptr(), a[1].dataset.tensors[0].data_ptr())

    def test_tiny_problem_can_be_learned(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        model = make_mlp(2, [16]); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        x = torch.tensor([[-1.,-1.],[1.,1.],[-1.,1.],[1.,-1.]])
        y = torch.tensor([0,1,0,1])
        for _ in range(100):
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(x), y)
            loss.backward(); optimizer.step()
        self.assertTrue(torch.equal(model(x).argmax(1), y))


if __name__ == '__main__': unittest.main()
