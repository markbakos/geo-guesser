from keras import layers, models, applications, regularizers

def create_hierarchical_model(num_regions=8, input_shape=(224, 224, 3)):
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers[:-50]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    shared = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)

    region = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.005))(shared)
    region = layers.BatchNormalization()(region)
    region = layers.Dropout(0.3)(region)
    region_output = layers.Dense(num_regions, activation='softmax', name='region')(region)

    coords = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.005))(shared)
    coords = layers.BatchNormalization()(coords)
    coords = layers.Dropout(0.3)(coords)
    coords = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005))(coords)
    coords = layers.BatchNormalization()(coords)
    coords = layers.Dropout(0.2)(coords)
    coordinate_output = layers.Dense(2, name='coordinates')(coords)

    model = models.Model(
        inputs=base_model.input,
        outputs=[region_output, coordinate_output]
    )

    return model