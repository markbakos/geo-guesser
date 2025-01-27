"use client"

import { useState } from "react"
import axios from "axios";
import Header from "@/app/components/Header"
import ImageUpload from "@/app/predict/ImageUpload"
import PredictionMap from "@/app/predict/PredictionMap"
import PredictionResult from "@/app/predict/PredictionResult"

interface ApiResponse {
    coordinates: [number, number]
    region: number
    region_confidence: number
}

export default function PredictPage() {
    const [prediction, setPrediction] = useState<[number, number] | null>(null)
    const [message, setMessage] = useState<string | null>(null)

    const handlePrediction = async (file: File) => {
        try {
            const formData = new FormData()
            formData.append('image', file)

            const response = await axios.post<ApiResponse>(
                    'http://127.0.0.1:8000/predict',
                    formData,
                    {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        }
                    }
                )

            const [longitude, latitude] = response.data.coordinates
            setPrediction([latitude, longitude])

            return response.data
        }
        catch (e) {
            setMessage("Prediction failed! Random coordinates generated. (Model not found / Server not running)")

            const randomLat = Math.random() * (90 - -90) + -90
            const randomLon = Math.random() * (180 - -180) + -180

            setPrediction([randomLat, randomLon])

            console.log(e)
        }
    }

    return (
        <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-grow">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <h1 className="text-3xl font-bold text-gray-900 mb-8">Predict Image Location</h1>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div>
                            <ImageUpload onUpload={handlePrediction} />
                            {prediction && <PredictionResult coordinates={prediction} />}
                            {message && <p className="px-4 text-red-700">{message}</p>}
                        </div>
                        <div className="h-96 md:h-auto">
                            <PredictionMap prediction={prediction} />
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}