from ilya_ezplot import plot_group, Metric

ms = Metric.load_all()

train_ms = [m for m in ms if 'train_loss' in m.name]
test_ms = [m for m in ms if 'test_loss' in m.name]

train_acc_ms = [m for m in ms if 'train_acc' in m.name]
test_acc_ms = [m for m in ms if 'test_acc' in m.name]

#
# plot_group(ms, smoothen=False, name='all_precise')
# plot_group(ms, name='all_smooth')


plot_group(train_ms, smoothen=False, name='train_precise')
plot_group(train_ms, name='train_smooth')


plot_group(test_ms, smoothen=False, name='test_precise')
plot_group(test_ms, name='test_smooth')

plot_group(train_acc_ms, smoothen=False, name='acc_train_precise')
plot_group(train_acc_ms, name='acc_train_smooth')

plot_group(test_acc_ms, smoothen=False, name='acc_test_precise')
plot_group(test_acc_ms, name='acc_test_smooth')