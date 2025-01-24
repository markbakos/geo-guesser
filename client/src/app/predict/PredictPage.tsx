"use client"

import { useState } from "react"
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

    const handlePrediction = async (file: File) => {
        await new Promise((resolve) => setTimeout(resolve, 2000))

        const response: ApiResponse = {
            coordinates: [67.05232238769531, 40.305145263671875],
            region: 2,
            region_confidence: 0.32080528140068054,
        }

        const [longitude, latitude] = response.coordinates
        setPrediction([latitude, longitude])
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