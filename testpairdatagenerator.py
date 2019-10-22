from pairdatagenerator import PairDataGenerator


file_path = 'data/epochs_4c'
file_template = '{}_BL5_ew32.h5'
sham_set = ['Sham102', 'Sham103', 'Sham104']
tbi_set = ['TBI101', 'TBI102', 'TBI103']
batch_size = 10
num_classes = 4
num_samples = 2048
dg = PairDataGenerator(file_path, file_template, sham_set, tbi_set, batch_size,
                       num_classes, num_samples, regenerate=True)
print(dg.indexes[0])
dg.on_epoch_end()
print(dg.indexes[0])
print(len(dg))
print(dg[0])
labels = dg.get_labels()
print(labels[0:3])