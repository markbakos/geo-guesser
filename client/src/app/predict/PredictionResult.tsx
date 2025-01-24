interface PredictionResultProps {
    coordinates: [number, number]
}

export default function PredictionResult({ coordinates }: PredictionResultProps) {
    const [latitude, longitude] = coordinates

    return (
        <div className="mt-8 p-4 bg-white shadow rounded-lg">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Prediction Result</h2>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <p className="text-sm font-medium text-gray-500">Latitude</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{latitude.toFixed(6)}</p>
                </div>
                <div>
                    <p className="text-sm font-medium text-gray-500">Longitude</p>
                    <p className="mt-1 text-lg font-semibold text-gray-900">{longitude.toFixed(6)}</p>
                </div>
            </div>
        </div>
    )
}