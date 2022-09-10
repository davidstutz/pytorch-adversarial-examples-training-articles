import os
import numpy
from common.log import log, LogLevel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.distribution import compressor
import altair


def rgb_html(rgb):
    rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return '#%02x%02x%02x' % rgb


def darken(rgb, alpha=0.9):
    rgb[0] = alpha*rgb[0]
    rgb[1] = alpha*rgb[1]
    rgb[2] = alpha*rgb[2]
    return rgb


def lighten(rgb, alpha=0.9):
    rgb[0] = 1 - (alpha * (1 - rgb[0]))
    rgb[1] = 1 - (alpha * (1 - rgb[1]))
    rgb[2] = 1 - (alpha * (1 - rgb[2]))
    return rgb


color_brewer_ = numpy.array([
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [245, 245, 145],
    # 10
    lighten([31, 120, 180]),
    lighten([251, 154, 153]),
    lighten([178, 223, 138]),
    lighten([51, 160, 44]),
    lighten([227, 26, 28]),
    lighten([253, 191, 111]),
    lighten([255, 127, 0]),
    lighten([202, 178, 214]),
    lighten([106, 61, 154]),
    lighten([255, 255, 153]),
    # 20
    lighten([31, 120, 180], 0.8),
    lighten([251, 154, 153], 0.8),
    lighten([178, 223, 138], 0.8),
    lighten([51, 160, 44], 0.8),
    lighten([227, 26, 28], 0.8),
    lighten([253, 191, 111], 0.8),
    lighten([255, 127, 0], 0.8),
    lighten([202, 178, 214], 0.8),
    lighten([106, 61, 154], 0.8),
    lighten([255, 255, 153], 0.8),
    # 30
    lighten([31, 120, 180], 0.7),
    lighten([251, 154, 153], 0.7),
    lighten([178, 223, 138], 0.7),
    lighten([51, 160, 44], 0.7),
    lighten([227, 26, 28], 0.7),
    lighten([253, 191, 111], 0.7),
    lighten([255, 127, 0], 0.7),
    lighten([202, 178, 214], 0.7),
    lighten([106, 61, 154], 0.7),
    lighten([255, 255, 153], 0.7),
    # 40
    lighten([31, 120, 180], 0.6),
    lighten([251, 154, 153], 0.6),
    lighten([178, 223, 138], 0.6),
    lighten([51, 160, 44], 0.6),
    lighten([227, 26, 28], 0.6),
    lighten([253, 191, 111], 0.6),
    lighten([255, 127, 0], 0.6),
    lighten([202, 178, 214], 0.6),
    lighten([106, 61, 154], 0.6),
    lighten([255, 255, 153], 0.6),
    # 50
    lighten([31, 120, 180], 0.5),
    lighten([251, 154, 153], 0.5),
    lighten([178, 223, 138], 0.5),
    lighten([51, 160, 44], 0.5),
    lighten([227, 26, 28], 0.5),
    lighten([253, 191, 111], 0.5),
    lighten([255, 127, 0], 0.5),
    lighten([202, 178, 214], 0.5),
    lighten([106, 61, 154], 0.5),
    lighten([255, 255, 153], 0.5),
], dtype=float)
color_brewer = color_brewer_/255.


BASIS_POINTS = [
    0,
    6.68,
    15.87,
    30.85,
    50.00,
    69.15,
    84.13,
    93.32,
    100.00
]


class Event:
    def __init__(self, step, tensor):
        self.step = step
        self.tensor = tensor


class TensorBoardAnalytics:
    def __init__(self, log_path):
        assert os.path.exists(log_path)
        assert os.path.isfile(log_path) or os.path.isdir(log_path)
        if os.path.isdir(log_path):
            assert len(os.listdir(log_path)) > 0
        self.event_acc = EventAccumulator(log_path, size_guidance={'tensors': 1000})

    def load(self):
        self.event_acc.Reload()

    def tags(self):
        return self.event_acc.Tags()

    def scalar(self, name):
        name = name.replace('.', '/')
        events = self.event_acc.Scalars(name)

        return [Event(e.step, numpy.array([e.value])) for e in events]

    def histogram(self, name):
        name = name.replace('.', '/')
        events = self.event_acc.Histograms(name)
        return [Event(e.step, numpy.array([e.histogram_value])) for e in events]

    def summarize(self, events):
        step = numpy.mean([e.step for e in events])
        values = numpy.sort([e.tensor for e in events])
        basis_points = numpy.percentile(values, BASIS_POINTS)

        return numpy.concatenate(([step], basis_points))

    def summarize_scalars(self, events, points=100):
        # Shrink to 100 histograms
        if points is None:
            interval = 1
        else:
            interval = max(1, len(events) // points)

        merged = []
        results = []
        for i, e in enumerate(events):
            if i > 0 and (i + 1) % interval == 0:
                results.append(self.summarize(merged))
                merged = []
            merged.append(e)
        if len(merged) > 0:
            results.append(self.summarize(merged))

        return numpy.array(results)

    def summarize_compressed_histogram(self, events):
        basis_points = [int(b) for b in numpy.multiply(BASIS_POINTS, 100)]
        results = []
        for e in events:
            buckets = compressor.compress_histogram(e.tensor)
            assert (len(buckets) == len(basis_points))
            for i, c in enumerate(buckets):
                assert (c.basis_point == basis_points[i])
            results.append([e.step] + [c.value for c in buckets])

        return numpy.asarray(results)


class AltairAnalytics:
    def __data_to_table(self, data):
        table = []

        for i in range(data.shape[0]):
            row = {'step': data[i, 0]}
            for j in range(1, 10):
                row['v%d' % j] = data[i, j]
            table.append(row)

        return altair.Data(values=table)

    def __render_density(self, table, name, line_color, range_color, levels, alpha, selection=None, x_scale=altair.Undefined, y_scale=altair.Undefined):
        areas = []
        for i in range(1, levels):
            y = 'v%d:Q' % (5 - i)
            y2 = 'v%d:Q' % (5 + i)

            areas.append(
                altair.Chart(table)
                    .mark_area(opacity=alpha ** i)
                    .encode(x=altair.X('step:Q', scale=x_scale), y=altair.Y(y, scale=y_scale), y2=altair.Y2(y2), color=altair.value(range_color))
            )

        # title=name
        line = altair.Chart(table).mark_line().encode(x=altair.X('step:Q', scale=x_scale), y=altair.Y("v5:Q", scale=y_scale, title=''), color=altair.value(line_color))

        if selection is not None:
            line = line.add_selection(selection)

        areas_sum = None
        for a in areas:
            if areas_sum is None:
                areas_sum = a
            else:
                areas_sum += a

        if areas_sum is None:
            return line
        else:
            return areas_sum + line

    def render_density(self, data, *, name, line_color, range_color, levels, alpha, height, width):
        table = self.__data_to_table(data)
        chart = self.__render_density(table, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha)
        chart = chart.properties(width=width, height=height)

        return chart

    def render_density_minimap(self, data, *, name, line_color, range_color, levels=5, alpha=0.6, height, width, height_minimap):
        table = self.__data_to_table(data)
        zoom = altair.selection_interval(encodings=["x", "y"])
        minimap = self.__render_density(table, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha, selection=zoom)
        detail = self.__render_density(table, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha,
                                       x_scale=altair.Scale(domain={'selection': zoom.name, "encoding": "x"}),
                                       y_scale=altair.Scale(domain={'selection': zoom.name, "encoding": "y"}))
        minimap = minimap.properties(width=width, height=height_minimap)
        detail = detail.properties(width=width, height=height)

        return detail & minimap

    def render_density_minimap_multiple(self, datas, *, names, levels=5, alpha=0.6, height, width, height_minimap):
        tables = [self.__data_to_table(d) for d in datas]

        zoom = altair.selection_interval(encodings=["x", "y"])

        minimaps = None
        for i, t in enumerate(tables):
            z = zoom if i == 0 else None
            minimap = self.__render_density(t, name=names[0], line_color=rgb_html(color_brewer_[i]),
                                            range_color=rgb_html(color_brewer_[i]), levels=levels, alpha=alpha, selection=z)
            if minimaps is None:
                minimaps = minimap
            else:
                minimaps += minimap

        details = None
        for i, t in enumerate(tables):
            detail = self.__render_density(t, name=names[i], line_color=rgb_html(color_brewer_[i]),
                                           range_color=rgb_html(color_brewer_[i]), levels=levels, alpha=alpha,
                                           x_scale=altair.Scale(domain={'selection': zoom.name, "encoding": "x"}),
                                           y_scale=altair.Scale(domain={'selection': zoom.name, "encoding": "y"}))
            if details is None:
                details = detail
            else:
                details += detail

        minimaps = minimaps.properties(width=width, height=height_minimap)
        details = details.properties(width=width, height=height)

        return details & minimaps


class AltairTensorBoardAnalytics(TensorBoardAnalytics, AltairAnalytics):
    def render_scalar(self, name, *, line_color='steelblue', range_color='steelblue', levels=5, alpha=0.6, points=100, height=400, width=800, height_minimap=100):
        if not name in self.tags()['scalars']:
            log('key %s not found in scalars' % name, LogLevel.ERROR)
            return
        data = self.summarize_scalars(self.scalar(name), points=points)
        return self.render_density_minimap(data, name=name, line_color=line_color, range_color=range_color,
                                           levels=levels, alpha=alpha, width=width, height=height, height_minimap=height_minimap)

    def render_scalar_multiple(self, names, *, levels=5, alpha=0.6, height=400, width=800, height_minimap=100):
        if len(names) == 0:
            log('no keys', LogLevel.ERROR)
            return
        for name in names:
            if not name in self.tags()['scalars']:
                log('key %s not found in scalars' % name, LogLevel.ERROR)
                return
        datas = [self.summarize_scalars(self.scalar(n)) for n in names]
        return self.render_density_minimap_multiple(datas, names=names, levels=levels, alpha=alpha, width=width, height=height, height_minimap=height_minimap)

    def render_histogram(self, name, *, line_color='steelblue', range_color='steelblue', levels=5, alpha=0.6, height=400, width=800, height_minimap=100):
        if not name in self.tags()['histograms']:
            log('key %s not found in histograms' % name, LogLevel.ERROR)
            return
        data = self.summarize_compressed_histogram(self.histogram(name))
        return self.render_density_minimap(data, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha,
                                           width=width, height=height, height_minimap=height_minimap)

    def render_histogram_multiple(self, names, *, levels=5, alpha=0.6, height=400, width=800, height_minimap=100):
        if len(names) == 0:
            log('no keys', LogLevel.ERROR)
            return
        for name in names:
            if not name in self.tags()['histograms']:
                log('key %s not found in histograms' % name, LogLevel.ERROR)
                return
        datas = [self.summarize_compressed_histogram(self.histogram(n)) for n in names]
        return self.render_density_minimap_multiple(datas, names=names, levels=levels, alpha=alpha, width=width, height=height, height_minimap=height_minimap)

    def render_scalar_simple(self, name, *, line_color='steelblue', range_color='steelblue', levels=5, alpha=0.6, points=100, height=400, width=800):
        if not name in self.tags()['scalars']:
            log('key %s not found in scalars' % name, LogLevel.ERROR)
            return
        data = self.summarize_scalars(self.scalar(name), points=points)
        return self.render_density(data, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha, width=width, height=height)

    def render_histogram_simple(self, name, *, line_color='steelblue', range_color='steelblue', levels=5, alpha=0.6, height=400, width=800):
        if not name in self.tags()['histograms']:
            log('key %s not found in histograms' % name, LogLevel.ERROR)
            return
        data = self.summarize_compressed_histogram(self.histogram(name))
        return self.render_density(data, name=name, line_color=line_color, range_color=range_color, levels=levels, alpha=alpha, width=width, height=height)
