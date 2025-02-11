import tensorflow as tf

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on the Earth."""
    R = 6371

    factor = tf.constant(3.141592653589793 / 180.0, dtype=tf.float32)
    lat1 = lat1 * factor
    lon1 = lon1 * factor
    lat2 = lat2 * factor
    lon2 = lon2 * factor

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = tf.math.sin(dlat / 2) ** 2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.asin(tf.math.sqrt(a))

    return R * c


def haversine_loss(y_true, y_pred):
    """Loss function using Haversine"""
    lat_true, lon_true = tf.unstack(y_true, axis=-1)
    lat_pred, lon_pred = tf.unstack(y_pred, axis=-1)

    return haversine_distance(lat_true, lon_true, lat_pred, lon_pred)


def location_accuracy(y_true, y_pred, threshold_km=30):
    """Calculate if a prediction is correct"""
    lat_true, lon_true = tf.unstack(y_true, axis=-1)
    lat_pred, lon_pred = tf.unstack(y_pred, axis=-1)

    distances = haversine_distance(lat_true, lon_true, lat_pred, lon_pred)
    return tf.reduce_mean(tf.cast(distances <= threshold_km, tf.float32))