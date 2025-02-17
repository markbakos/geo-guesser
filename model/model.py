from keras import layers, models, applications, regularizers

def create_model(num_cities=5, input_shape=(224, 224, 3)):
    """Create a deep learning model with EfficientNetV2S backbone for location prediction."""
    base_model = applications.EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )

    base_model.get_layer('top_conv').name = 'top_conv'

    for layer in base_model.layers:
        layer.trainable = False

    x1 = layers.GlobalAveragePooling2D()(base_model.output)
    x2 = layers.GlobalMaxPooling2D()(base_model.output)
    x = layers.Concatenate()([x1, x2])

    shared = layers.Dense(768, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.5)(shared)

    shared_residual = layers.Dense(768, activation='relu')(shared)
    shared_residual = layers.BatchNormalization()(shared_residual)
    shared_residual = layers.Dropout(0.5)(shared_residual)
    shared = layers.Add()([shared, shared_residual])

    city = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared)
    city = layers.BatchNormalization()(city)
    city = layers.Dropout(0.4)(city)
    city_output = layers.Dense(num_cities, activation='softmax', name='city')(city)

    coords = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared)
    coords = layers.BatchNormalization()(coords)
    coords = layers.Dropout(0.4)(coords)
    coordinate_output = layers.Dense(2, activation='linear', name='coordinates')(coords)

    model = models.Model(
        inputs=base_model.input,
        outputs=[city_output, coordinate_output]
    )

    return model