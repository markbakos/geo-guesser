interface PredictionResultProps {
    coordinates: {
        latitude: number
        longitude: number
    }
    city: string
    city_confidence: number
}

export default function PredictionResult({ coordinates, city, city_confidence }: PredictionResultProps) {

    return (
        <div className="mt-8 p-4 bg-white shadow rounded-lg">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Prediction Result</h2>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <p className="text-sm font-medium text-gray-500">City</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{city}</p>
                </div>
                <div>
                    <p className="text-sm font-medium text-gray-500">City Confidence</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{(city_confidence ?? 0).toFixed(6)}</p>
                </div>
                <div>
                    <p className="text-sm font-medium text-gray-500">Latitude</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{coordinates.latitude.toFixed(6)}</p>
                </div>
                <div>
                    <p className="text-sm font-medium text-gray-500">Longitude</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{coordinates.longitude.toFixed(6)}</p>
                </div>
            </div>
        </div>
    )
}