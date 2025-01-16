from keras import layers, models, applications, regularizers


def create_hierarchical_model(num_regions=8, num_scenes=2, input_shape=(224, 224, 3)):
    base_model = applications.EfficientNetV2L(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers:
        layer.trainable = False

    x1 = layers.GlobalAveragePooling2D()(base_model.output)
    x2 = layers.GlobalMaxPooling2D()(base_model.output)
    x = layers.Concatenate()([x1, x2])

    shared = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.5)(shared)

    shared_residual = layers.Dense(1024, activation='relu')(shared)
    shared_residual = layers.BatchNormalization()(shared_residual)
    shared_residual = layers.Dropout(0.5)(shared_residual)
    shared = layers.Add()([shared, shared_residual])

    region = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared)
    region = layers.BatchNormalization()(region)
    region = layers.Dropout(0.4)(region)
    region_output = layers.Dense(num_regions, activation='softmax', name='region')(region)

    scene = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared)
    scene = layers.BatchNormalization()(scene)
    scene = layers.Dropout(0.4)(scene)
    scene_output = layers.Dense(num_scenes, activation='softmax', name='scene')(scene)

    region_features = layers.Dense(256, activation='relu')(region)
    scene_features = layers.Dense(256, activation='relu')(scene)

    combined_features = layers.Concatenate()([region_features, scene_features])

    coords = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined_features)
    coords = layers.BatchNormalization()(coords)
    coords = layers.Dropout(0.4)(coords)

    latitude = layers.Dense(1, activation='linear', name='latitude')(coords)
    longitude = layers.Dense(1, activation='linear', name='longitude')(coords)
    coordinate_output = layers.Concatenate(name='coordinates')([latitude, longitude])

    model = models.Model(
        inputs=base_model.input,
        outputs=[region_output, scene_output, coordinate_output]
    )

    return model