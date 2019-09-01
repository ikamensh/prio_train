from ilya_ezplot import plot_group, Metric

all_metrics = Metric.load_all()

def plot_2_ways(filter: str):
    metrics = [m for m in all_metrics if filter in m.name]

    unique_names = {m.name.split('$')[0] for m in metrics}
    grouped_ms = []
    for name in unique_names:
        ms = [m for m in metrics if m.name == name]
        grouped_ms.append(sum(ms))

    plot_group(grouped_ms, smoothen=False, name=f'{filter}_precise')
    plot_group(grouped_ms, name=f'{filter}_smooth')

plot_2_ways('train_loss')
plot_2_ways('test_loss')
plot_2_ways('train_acc')
plot_2_ways('test_acc')