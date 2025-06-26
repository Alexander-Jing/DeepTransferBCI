# -*- coding: utf-8 -*-
import torch
import contextlib
import json
import time
import pprint
import os
from typing import Any, Dict, Generator
from io import StringIO
from contextlib import contextmanager
import numpy as np



task2metrics = {"classification": ["cross_entropy", "accuracy_top1"]}
auxiliary_metrics_dict = {
    "preadapted_cross_entropy": "cross_entropy",
    "preadapted_accuracy_top1": "accuracy_top1",
}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""

    def __init__(self, metrics_to_track):
        self.metrics_to_track = metrics_to_track
        self.reset()

    def reset(self):
        self.stat = dict((name, AverageMeter()) for name in self.metrics_to_track)

    def add_stat(self, metric_name: str):
        self.stat[metric_name] = AverageMeter()

    def get_metrics_performance(self):
        return [self.stat[metric].avg for metric in self.metrics_to_track]

    def update_metrics(self, metric_stat, n_samples):
        for name, value in metric_stat.items():
            self.stat[name].update(value, n_samples)

    def __call__(self):
        return dict((name, val.avg) for name, val in self.stat.items())

    def get_current_val(self):
        return dict((name, val.val) for name, val in self.stat.items())

    def get_val_by_name(self, metric_name):
        return dict([(metric_name, self.stat[metric_name].val)])

class Metrics(object):
    def __init__(self, scenario) -> None:
        self._conf = scenario
        self._init_metrics()

    def _init_metrics(self) -> None:
        self._metrics = task2metrics[self._conf.task]
        self.tracker = RuntimeTracker(metrics_to_track=self._metrics)
        self._primary_metrics = self._metrics[0]

    def init_auxiliary_metric(self, metric_name: str):
        self._metrics.append(metric_name)
        self.tracker.add_stat(metric_name)

    @torch.no_grad()
    def eval(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        results = dict()
        for metric_name in self._metrics:
            if not metric_name in auxiliary_metrics_dict.keys():
                results[metric_name] = eval(metric_name)(y, y_hat)
            else:
                continue
        self.tracker.update_metrics(results, n_samples=y.size(0))
        return results

    @torch.no_grad()
    def eval_auxiliary_metric(
        self, y: torch.Tensor, y_hat: torch.Tensor, metric_name: str
    ):
        assert (
            metric_name in self._metrics
        ), "The target metric must be in the list of metrics."
        results = dict()
        results[metric_name] = eval(auxiliary_metrics_dict[metric_name])(y, y_hat)
        self.tracker.update_metrics(results, n_samples=y.size(0))
        return results


"""list some common metrics."""


def _accuracy(target, output, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size).item()


def accuracy_top1(target, output, topk=1):
    """Computes the precision@k for the specified values of k"""
    return _accuracy(target, output, topk)


def accuracy_top5(target, output, topk=5):
    """Computes the precision@k for the specified values of k"""
    return _accuracy(target, output, topk)


cross_entropy_loss = torch.nn.CrossEntropyLoss()


def cross_entropy(target, output):
    """Cross entropy loss"""
    return cross_entropy_loss(output, target).item()

@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield



class Logger(object):
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.folder_path = folder_path
        self.json_file_path = os.path.join(folder_path, "log-1.json")
        self.txt_file_path = os.path.join(folder_path, "log.txt")
        self.values = []
        self.pp = MyPrettyPrinter(indent=2, depth=3, compact=True)

    def log_metric(
        self,
        name: str,
        values: Dict[str, Any],
        tags: Dict[str, Any],
        display: bool = False,
    ) -> None:
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})

        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def pretty_print(self, value: Any) -> None:
        self.pp.pprint(value)

    def log(self, value: str, display: bool = True) -> None:
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        if display:
            print(content)
        self.save_txt(content)

    def save_json(self) -> None:
        """Save the internal memory to a file."""
        with open(self.json_file_path, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        if len(self.values) > 1e4:
            # reset 'values' and redirect the json file to a different path.
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value: str) -> None:
        with open(self.txt_file_path, "a") as f:
            f.write(value + "\n")

    def redirect_new_json(self) -> None:
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(self.folder_path) if "json" in file
        ]
        self.json_file_path = os.path.join(
            self.folder_path, "log-{}.json".format(len(existing_json_files) + 1)
        )

class MyPrettyPrinter(pprint.PrettyPrinter):
    """Borrowed from
    https://stackoverflow.com/questions/30062384/pretty-print-namedtuple
    """

    def format_namedtuple(self, object, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + "(")
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            self.format_namedtuple_items(
                object_dict.items(),
                inline_stream,
                indent,
                allowance + 1,
                context,
                level,
                inline=True,
            )
            max_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > max_width:
                self.format_namedtuple_items(
                    object_dict.items(),
                    stream,
                    indent,
                    allowance + 1,
                    context,
                    level,
                    inline=False,
                )
            else:
                stream.write(inline_stream.getvalue())
        write(")")

    def format_namedtuple_items(
        self, items, stream, indent, allowance, context, level, inline=False
    ):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ", "
        else:
            delimnl = ",\n" + " " * indent
            write("\n" + " " * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + "=")
            self._format(
                ent,
                stream,
                indent + len(key) + 2,
                allowance if last else 1,
                context,
                level,
            )
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if hasattr(object, "_asdict") and type(object).__repr__ not in self._dispatch:
            self._dispatch[type(object).__repr__] = MyPrettyPrinter.format_namedtuple
        super()._format(object, stream, indent, allowance, context, level)


class Timer(object):
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:

    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(
        self,
        device: str,
        verbosity_level: int = 1,
        log_fn=None,
        skip_first: bool = True,
        on_cuda: bool = True,
    ) -> None:
        self.device = device
        self.verbosity_level = verbosity_level
        self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first
        self.cuda_available = torch.cuda.is_available() and on_cuda

        self.reset()

    def reset(self) -> None:
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(
        self, label: str, step: int = -1, epoch: int = -1.0, verbosity: int = 1
    ) -> Generator[None, None, None]:
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time()
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if label not in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if label not in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif label not in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        if self.call_counts[label] > 0:
            # We will reduce the probability of logging a timing
            # linearly with the number of time we have seen it.
            # It will always be recorded in the totals, though.
            if np.random.rand() < 1 / self.call_counts[label]:
                self.log_fn(
                    "timer",
                    {"step": step, "epoch": epoch, "value": end - start},
                    {"event": label},
                )

    def summary(self) -> None:
        """
        Return a summary in string-form of all the timings recorded so far
        """
        if len(self.totals) > 0:
            with StringIO() as buffer:
                total_avg_time = 0
                print("--- Timer summary ------------------------", file=buffer)
                print("  Event   |  Count | Average time |  Frac.", file=buffer)
                for event_label in sorted(self.totals):
                    total = self.totals[event_label]
                    count = self.call_counts[event_label]
                    if count == 0:
                        continue
                    avg_duration = total / count
                    total_runtime = (
                        self.last_time[event_label] - self.first_time[event_label]
                    )
                    runtime_percentage = 100 * total / total_runtime
                    total_avg_time += avg_duration if "." not in event_label else 0
                    print(
                        f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                        file=buffer,
                    )
                print("-------------------------------------------", file=buffer)
                event_label = "total_averaged_time"
                print(
                    f"- {event_label:30s}| {count:6d} | {total_avg_time:11.5f}s |",
                    file=buffer,
                )
                print("-------------------------------------------", file=buffer)
                return buffer.getvalue()

    def _cuda_sync(self) -> None:
        """Finish all asynchronous GPU computations to get correct timings"""
        if self.cuda_available:
            torch.cuda.synchronize(device=self.device)

    def _default_log_fn(self, _: Any, values: Dict, tags: Dict) -> None:
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")

