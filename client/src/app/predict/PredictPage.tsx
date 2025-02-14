"use client"

import { useState } from "react"
import axios from "axios";
import Header from "@/app/components/Header"
import ImageUpload from "@/app/predict/ImageUpload"
import PredictionMap from "@/app/predict/PredictionMap"
import PredictionResult from "@/app/predict/PredictionResult"

interface ApiResponse {
    coordinates: {
        latitude: number
        longitude: number
    }
    city: string
    city_confidence: number
}

export default function PredictPage() {
    const [prediction, setPrediction] = useState<ApiResponse | null>(null)
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


            setMessage("")
            setPrediction({
                coordinates: response.data.coordinates,
                city: response.data.city,
                city_confidence: response.data.city_confidence
            })

            console.log(prediction)

            return response.data
        }
        catch (e) {
            setMessage("Prediction failed! Random coordinates generated.")

            const cities= [
                "Budapest",
                "Cairo",
                "Ottawa",
                "Tokyo",
                "Canberra"
            ]

            const randomLat = Math.random() * (90 - -90) + -90
            const randomLon = Math.random() * (180 - -180) + -180

            setPrediction({
                coordinates: {
                    latitude: randomLat,
                    longitude: randomLon
                },
                city: cities[Math.floor(Math.random() * cities.length)],
                city_confidence: Math.random()
            })

            console.log(e)
        }
    }

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            <Header />
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <h1 className="text-4xl font-extrabold mb-1 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
                    Predict Image Location
                </h1>
                <p className="text-md text-center mb-8">
                    The model is not hosted yet, to use this model you have to start the server yourself.
                </p>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="space-y-8">
                        <ImageUpload onUpload={handlePrediction}/>
                        {prediction && <PredictionResult coordinates={prediction.coordinates} city={prediction.city} city_confidence={prediction.city_confidence}/>}

                    </div>
                    <div>
                        <PredictionMap prediction={prediction ? prediction.coordinates : null}/>
                    </div>
                </div>
                <div className="mt-3">
                    {message && <p className="text-xl text-red-500 font-semibold">{message}</p>}
                </div>
            </main>
        </div>
    )
}